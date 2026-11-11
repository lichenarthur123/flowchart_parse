"""识别各节点的外框，并将局部框坐标还原到高清图。"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from entity_extraction import save_json
from model_client import ModelClient
from node_geometry import local_to_global_box, mark_target, point_box_distance
from text_extraction import recognize_node_text


POINT_TOLERANCE_PX = 8


def crop_around_point(image: Image.Image, point: list[float], size: int) -> tuple[Image.Image, list[int], list[float]]:
    """原尺寸裁图，同时返回截图的全局范围和局部点坐标。"""
    width, height = min(size, image.width), min(size, image.height)
    left = max(0, min(image.width - width, math.floor(point[0] - width / 2)))
    top = max(0, min(image.height - height, math.floor(point[1] - height / 2)))
    bbox = [left, top, left + width, top + height]
    return image.crop(tuple(bbox)), bbox, [point[0] - left, point[1] - top]


def draw_global_boxes(image: Image.Image, nodes: list[dict], output: Path) -> None:
    """在高清原图上绘制外框和节点 ID。"""
    result = image.copy()
    draw = ImageDraw.Draw(result)
    line = max(3, round(max(result.size) / 2000))
    try:
        font = ImageFont.load_default(size=max(20, line * 3))
    except TypeError:
        font = ImageFont.load_default()
    for node in nodes:
        x0, y0, x1, y1 = node['bbox_global_px']
        color = 'orange' if node.get('review_reason') else 'red'
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line)
        draw.text((x0 + line, max(0, y0 - line * 3)), node['id'].replace('node_', ''),
                  fill=color, font=font, stroke_width=1, stroke_fill='white')
    result.save(output)


def process_node_boxes(client: ModelClient, image_path: Path, output: Path, nodes_path: Path,
                       crop_size: int = 1200, resume: bool = False,
                       ocr_client: ModelClient | None = None, workers: int = 4) -> None:
    """逐节点定位局部框，识别框内文字并保存全局框。"""
    if crop_size <= 0:
        raise ValueError('节点框识别的截图尺寸必须大于 0。')
    data = json.loads(nodes_path.read_text(encoding='utf-8'))
    nodes = data.get('nodes')
    if data.get('coordinate_system') != 'pixels_top_left' or not isinstance(nodes, list):
        raise ValueError('节点文件须包含左上角为原点的像素坐标和 nodes 数组。')
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    seen = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get('id'), str) or not node['id'] or node['id'] in seen:
            raise ValueError('节点 ID 缺失或重复。')
        seen.add(node['id'])
        point = node.get('center_global_px')
        if not isinstance(point, list) or len(point) != 2 or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in point) or not (0 <= point[0] < image.width and 0 <= point[1] < image.height):
            raise ValueError(f"节点 {node['id']} 的全局坐标无效。")
    crop_dir = output / 'node_crops'
    crop_dir.mkdir(exist_ok=True)
    previous_nodes = {}
    previous_boxes = output / 'boxes.json'
    if resume and ocr_client is None and previous_boxes.is_file():
        previous_data = json.loads(previous_boxes.read_text(encoding='utf-8'))
        previous_nodes = {n['id']: n for n in previous_data.get('nodes', []) if isinstance(n, dict) and 'id' in n}

    def process_one(index: int, node: dict):
        folder = crop_dir / f'{index:04d}'
        folder.mkdir(exist_ok=True)
        result_path = folder / 'result.json'
        crop, crop_bbox, local_point = crop_around_point(image, node['center_global_px'], crop_size)
        marked_path = folder / 'marked.png'
        mark_target(crop, local_point)
        crop.save(marked_path)
        request = {'id': node['id'], 'center_global_px': node['center_global_px'],
                   'crop_global_px': crop_bbox, 'center_local_px': local_point}
        response = None
        if resume and result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            if saved.get('request') != request:
                raise ValueError(f"节点 {node['id']} 的缓存与当前截图不匹配。")
            response = saved['response']
        if response is None:
            prompt = (f'高清截图宽 {crop.width}、高 {crop.height}。红点标记目标节点，'
                      '只返回其最小水平外接框的截图像素坐标 {"bbox":[x0,y0,x1,y1]}。'
                      '不要包含文字、连线或其他节点；粗横条需框住整体。无法确定返回 {"bbox":null}。')
            print(f'节点框 {index}/{len(nodes)}：{node["id"]}', flush=True)
            response = client.ask_json(prompt, [marked_path], stage='node_bounding_box')
            save_json(result_path, {'request': request, 'response': response})
        local_box = response.get('bbox')
        global_box = local_to_global_box(local_box, crop_bbox, local_point, POINT_TOLERANCE_PX)
        entry = {'id': node['id'], 'type': node.get('type', 'unknown'),
                 'center_global_px': node['center_global_px']}
        if global_box is None:
            return None, {**entry, 'reason': 'missing_or_invalid_local_box', 'crop': folder.name}
        entry['bbox_global_px'] = global_box
        if ocr_client is not None:
            print(f'节点文字 OCR {index}/{len(nodes)}：{node["id"]}', flush=True)
            entry.update(recognize_node_text(ocr_client, image, global_box, folder / 'ocr', resume))
        else:
            previous = previous_nodes.get(node['id'], {})
            if (previous.get('bbox_global_px') == global_box and
                    previous.get('center_global_px') == node['center_global_px'] and
                    previous.get('text_status') in ('recognized', 'empty')):
                entry.update(text=previous.get('text'), text_status=previous['text_status'])
            else:
                entry.update(text=None, text_status='not_requested')
        distance = point_box_distance(local_point, local_box)
        if distance:
            entry.update(review_reason='point_outside_box', point_distance_px=distance)
        return entry, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda item: process_one(*item), enumerate(nodes, 1)))
    found = [entry for entry, _ in results if entry is not None]
    unresolved = [item for _, item in results if item is not None]
    draw_global_boxes(image, found, output / 'global_boxes_marked.png')
    save_json(output / 'boxes.json', {'coordinate_system': 'pixels_top_left', 'nodes': found})
    save_json(output / 'unresolved_boxes.json', {'nodes': unresolved})
    near = sum('review_reason' in node for node in found)
    print(f'节点框完成：成功 {len(found)}（其中 {near} 个点在框外需复核），未确定 {len(unresolved)}。', flush=True)
