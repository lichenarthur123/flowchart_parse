"""识别节点框内文字并缓存结果。"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from PIL import Image

from model_client import ModelClient

NODE_PROMPT = 'Please output only the text content from the image without any additional descriptions or formatting.'
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
        lines = []
        for line in raw.splitlines():
            match = re.match(r'^\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){4}\s*,\s*(.+)$', line)
            lines.append(match.group(1).strip() if match else line.strip())
        text = '\n'.join(line for line in lines if line)
    if text in ('无文字', '无文本', '没有文字', '没有文本', '无', '[]'):
        text = ''
    return {'text': text, 'text_status': 'recognized' if text else 'empty'}
