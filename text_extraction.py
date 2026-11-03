"""OCR node text and associate positioned text with edge paths."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from model_client import ModelClient

NODE_PROMPT = 'Please output only the text content from the image without any additional descriptions or formatting.'
EDGE_PROMPT = ('请检测识别所有文字，并逐项返回文字内容和坐标，输出 JSON 数组，'
               '如 [{"text":"内容","rotate_rect":[x,y,w,h,angle]}]。没有文字返回 []。')
VERSION = 1


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _model_id(client: ModelClient) -> dict:
    config = client.config
    return {'base_url': config.base_url, 'model': config.model,
            'max_tokens': config.max_tokens, 'enable_thinking': config.enable_thinking}


def _ask_cached(client: ModelClient, image: Image.Image, folder: Path, prompt: str, resume: bool,
                stage: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / 'ocr_input.png'
    image.save(image_path)
    request = {'version': VERSION, 'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
               'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'model': _model_id(client)}
    result_path = folder / 'ocr_result.json'
    if resume and result_path.is_file():
        previous = json.loads(result_path.read_text(encoding='utf-8'))
        if previous.get('request') == request and isinstance(previous.get('text'), str):
            return previous['text']
    result = client.ask_text(prompt, [image_path], stage=stage)
    _write_json(result_path, {'request': request, 'text': result})
    return result


def _decode(raw: str):
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1]).strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def recognize_node_text(client: ModelClient, image: Image.Image, box: list[float],
                        folder: Path, resume: bool = False) -> dict:
    """Recognize unmarked pixels inside the node's model-provided box."""
    bounds = (max(0, math.floor(box[0])), max(0, math.floor(box[1])),
              min(image.width, math.ceil(box[2])), min(image.height, math.ceil(box[3])))
    crop = image.crop(bounds)
    raw = _ask_cached(client, crop, folder, NODE_PROMPT, resume, 'node_ocr').strip()
    data = _decode(raw)
    if isinstance(data, list):
        text = '\n'.join(str(item['text']).strip() for item in data
                         if isinstance(item, dict) and isinstance(item.get('text'), str) and item['text'].strip())
    elif isinstance(data, dict) and isinstance(data.get('text'), str):
        text = data['text'].strip()
    elif data is not None:
        text = ''
    else:
        # Some OCR endpoints return one CSV line per recognition.
        lines = []
        for line in raw.splitlines():
            match = re.match(r'^\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){4}\s*,\s*(.+)$', line)
            lines.append(match.group(1).strip() if match else line.strip())
        text = '\n'.join(line for line in lines if line)
    if text in ('无文字', '无文本', '没有文字', '没有文本', '无', '[]'):
        text = ''
    return {'text': text, 'text_status': 'recognized' if text else 'empty'}


def _starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, size - overlap))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def _ocr_box(item: dict) -> list[float] | None:
    rect = item.get('rotate_rect')
    if isinstance(rect, list) and len(rect) == 5 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in rect):
        cx, cy, w, h, angle = rect
        theta = math.radians(angle)
        width = abs(w*math.cos(theta)) + abs(h*math.sin(theta))
        height = abs(w*math.sin(theta)) + abs(h*math.cos(theta))
        return [cx-width/2, cy-height/2, cx+width/2, cy+height/2]
    bbox = item.get('bbox', item.get('bbox_2d'))
    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in bbox):
        return [float(x) for x in bbox]
    return None


def _items(raw: str) -> list[dict]:
    data = _decode(raw)
    if isinstance(data, dict):
        data = data.get('texts', data.get('items', data.get('words_info', [data] if isinstance(data.get('text'), str) else [])))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict) and isinstance(item.get('text'), str) and item['text'].strip()]
    # Keep unlocated text visible instead of inventing coordinates.
    text = raw.strip()
    if text in ('', '[]', '无文字', '无文本', '没有文字', '没有文本'):
        return []
    return [{'text': line.strip()} for line in text.splitlines() if line.strip()]


def _iou(a: list[float], b: list[float]) -> float:
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0


def _same_text_box(first: dict, second: dict) -> bool:
    normalize = lambda text: ''.join(unicodedata.normalize('NFKC', text).split())
    if normalize(first['text']) != normalize(second['text']):
        return False
    a, b = first['bbox_global_px'], second['bbox_global_px']
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    small_area = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return _iou(a, b) >= 0.4 or (small_area > 0 and overlap/small_area >= 0.6)


def _point_segment(point, a, b) -> float:
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx + dy*dy
    t = max(0.0, min(1.0, ((point[0]-a[0])*dx + (point[1]-a[1])*dy)/length2)) if length2 else 0
    return math.hypot(point[0]-a[0]-t*dx, point[1]-a[1]-t*dy)


def _box_path_distance(box: list[float], path: list[list[float]]) -> float:
    x1, y1, x2, y2 = box
    corners = [(x1,y1), (x1,y2), (x2,y1), (x2,y2)]
    rectangle = (math.floor(x1), math.floor(y1), max(1, math.ceil(x2-x1)), max(1, math.ceil(y2-y1)))
    distances = []
    for a,b in zip(path, path[1:]):
        if cv2.clipLine(rectangle, tuple(round(x) for x in a), tuple(round(x) for x in b))[0]:
            return 0.0
        for x,y in (a,b):
            distances.append(math.hypot(max(x1-x, 0, x-x2), max(y1-y, 0, y-y2)))
        distances.extend(_point_segment(corner, a, b) for corner in corners)
    return min(distances, default=float('inf'))


def extract_edge_text(client: ModelClient, image: Image.Image, graph: dict, output: Path,
                      resume: bool = False, tile_size: int = 1200, overlap: int = 200,
                      max_distance: float = 75, ambiguity_margin: float = 15,
                      region: list[int] | None = None) -> tuple[dict, dict]:
    """Read outside-node text in overlapping tiles, then associate by geometry.

    Ambiguous or unlocated text remains separate. Paths on shared trunks can
    legitimately make ownership ambiguous; do not select one arbitrarily.
    """
    if tile_size <= overlap or overlap < 0:
        raise ValueError('OCR 切片尺寸和重叠无效。')
    masked = image.convert('RGB')
    draw = ImageDraw.Draw(masked)
    for node in graph['nodes']:
        draw.rectangle(tuple(round(v) for v in node['bbox_global_px']), fill='white')
    folder = output.parent / 'edge_ocr_cache'
    folder.mkdir(parents=True, exist_ok=True)
    blocks, unlocated = [], []
    x0,y0,x1,y1 = region or [0, 0, image.width, image.height]
    for row, offset_y in enumerate(_starts(y1-y0, tile_size, overlap), 1):
        for col, offset_x in enumerate(_starts(x1-x0, tile_size, overlap), 1):
            x, y = x0+offset_x, y0+offset_y
            right, bottom = min(x+tile_size, x1), min(y+tile_size, y1)
            tile = masked.crop((x,y,right,bottom))
            # Both common OCR conventions (pixels and normalized 0..1000)
            # coincide on a 1000x1000 input. Letterbox retains aspect ratio.
            scale = 1000/max(tile.size)
            scaled = tile.resize((max(1,round(tile.width*scale)), max(1,round(tile.height*scale))), Image.Resampling.LANCZOS)
            square = Image.new('RGB', (1000,1000), 'white')
            square.paste(scaled, (0,0))
            name = f'r{row:03d}_c{col:03d}'
            print(f'边文字 OCR {name}', flush=True)
            raw = _ask_cached(client, square, folder / name, EDGE_PROMPT, resume, 'edge_ocr')
            for item in _items(raw):
                text = item['text'].strip()
                box = _ocr_box(item)
                if not box or not (0 <= box[0] < box[2] <= scaled.width+2 and 0 <= box[1] < box[3] <= scaled.height+2):
                    unlocated.append({'text': text, 'tile': name, 'reason': 'missing_or_invalid_text_box'})
                    continue
                bbox = [round(max(x, x+box[0]/scale), 1), round(max(y, y+box[1]/scale), 1),
                        round(min(right, x+box[2]/scale), 1), round(min(bottom, y+box[3]/scale), 1)]
                if any(_iou(bbox, node['bbox_global_px']) > 0.1 for node in graph['nodes']):
                    continue
                duplicate = next((b for b in blocks if _same_text_box(b, {'text': text, 'bbox_global_px': bbox})), None)
                if duplicate is None:
                    blocks.append({'text': text, 'bbox_global_px': bbox})
    result = {**graph, 'edges': [{**edge, 'text': '', 'text_status': 'none_detected', 'text_blocks': []} for edge in graph['edges']]}
    assigned, uncertain = [], list(unlocated)
    for index, block in enumerate(sorted(blocks, key=lambda b:(b['bbox_global_px'][1], b['bbox_global_px'][0])), 1):
        block['id'] = f'text_{index:04d}'
        if re.match(r'^\s*(注|备注|说明)\s*[:：]', block['text']):
            uncertain.append({**block, 'reason': 'annotation'})
            continue
        candidates = sorted((_box_path_distance(block['bbox_global_px'], edge['path_global_px']), edge['id'])
                            for edge in result['edges'])
        close = [(dist, edge_id) for dist,edge_id in candidates if dist <= max_distance]
        if not close:
            uncertain.append({**block, 'reason': 'no_nearby_edge'})
        elif len(close) > 1 and close[1][0] - close[0][0] <= ambiguity_margin:
            ambiguous_ids = {edge_id for dist,edge_id in close if dist-close[0][0] <= ambiguity_margin}
            for edge in result['edges']:
                if edge['id'] in ambiguous_ids:
                    edge['text_status'] = 'unresolved'
            uncertain.append({**block, 'reason': 'ambiguous_edge',
                              'candidates': [{'edge_id': edge_id, 'distance_px': round(dist,1)}
                                             for dist,edge_id in close if dist-close[0][0] <= ambiguity_margin]})
        else:
            target = next(edge for edge in result['edges'] if edge['id'] == close[0][1])
            target['text_blocks'].append(block)
            if target['text_status'] != 'unresolved':
                target['text_status'] = 'recognized'
            assigned.append({**block, 'edge_id': target['id'], 'distance_px': round(close[0][0],1)})
    for edge in result['edges']:
        edge['text_blocks'].sort(key=lambda b: (b['bbox_global_px'][1],b['bbox_global_px'][0]))
        edge['text'] = '\n'.join(b['text'] for b in edge['text_blocks'])
    return result, {'coordinate_system':'pixels_top_left', 'assigned':assigned, 'unresolved':uncertain,
                    'note':'文字与线条按几何距离关联；不确定的关联未写入边。'}
