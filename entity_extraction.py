"""Extract node entities and convert crop coordinates to full-resolution coordinates."""
from __future__ import annotations

import json
import hashlib
import math
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from model_client import ModelClient
from node_geometry import mark_target, local_to_global_box


def extract_region_entities(client: ModelClient, region_path: Path, region: dict) -> list[dict]:
    """一次识别节点和局部中心点，再由程序换算高清图坐标。"""
    width, height = region["width"], region["height"]
    prompt = f'''请识别流程图中实际可见的所有节点，并给出每个节点的类型和中心点坐标。
节点包括开始或结束圆形、过程矩形、判断菱形、汇合小菱形、并行分叉或同步横条。
只识别节点形状，不要识别或输出节点内的文字、编号、旁注、箭头或连接关系。
不要把标题、背景、普通旁注、箭头和连接线当成节点；不要补充不存在的节点。
按节点中心从上到下、从左到右排序，同一节点只列一次。类型无法确定时填 unknown。
图片宽 {width} 像素、高 {height} 像素，原点为左上角，x 向右、y 向下。
中心点必须使用本图片的像素坐标：x 为 0 到 {width-1}，y 为 0 到 {height-1}，不要归一化，不要输出框。
每个点应位于对应节点形状的几何中心，不要标在节点外的空白、箭头或连接线上。
节点部分被截断或中心位置无法确定时，x 和 y 都填 null，不要猜测。
只返回 JSON 对象：{{"nodes":[{{"type":"process","x":123,"y":456}}]}}。
type 只能是 start、end、process、decision、merge、fork、join、unknown。'''
    response = client.ask_json(prompt, [region_path], stage='node_location')
    nodes = response.get("nodes")
    if not isinstance(nodes, list) or any(not isinstance(item, dict) for item in nodes):
        raise ValueError("节点识别结果缺少有效的 nodes 列表。")
    x0, y0, _, _ = region["source_bbox"]
    scale_x, scale_y = region["scale_to_global"]
    valid_types = {"start", "end", "process", "decision", "merge", "fork", "join", "unknown"}
    result = []
    for index, node in enumerate(nodes, 1):
        x, y = node.get("x"), node.get("y")
        valid = (isinstance(x, (int, float)) and not isinstance(x, bool)
                 and isinstance(y, (int, float)) and not isinstance(y, bool)
                 and 0 <= x < width and 0 <= y < height)
        node_type = node.get("type")
        if not isinstance(node_type, str) or node_type not in valid_types:
            node_type = "unknown"
        result.append({"id": f"node_{index:03d}", "type": node_type,
                       "center_local_px": [x, y] if valid else None,
                       "center_global_px": [x0 + x * scale_x, y0 + y * scale_y] if valid else None,
                       "status": "located_unverified" if valid else "unresolved"})
    return result


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_region(image_path: Path | Image.Image, bbox: tuple[int, int, int, int], output: Path,
                   max_size: int = 2000) -> dict:
    """从高清图原尺寸裁剪；局部图不缩放。"""
    x0, y0, x1, y1 = bbox
    if max_size <= 0:
        raise ValueError("局部图最大边长必须大于 0。")
    source = nullcontext(image_path) if isinstance(image_path, Image.Image) else Image.open(image_path)
    with source as image:
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError("局部图范围无效或超出高清图边界。")
        region = image.crop(bbox)
        if max(region.size) > max_size:
            raise ValueError(f"局部图 {region.width}×{region.height} 超过 --tile-max-size；请增加等分数或调整限制。")
        region.save(output / "region.png")
        return {"source_bbox": list(bbox), "width": region.width, "height": region.height,
                "scale_to_global": [(x1 - x0) / region.width, (y1 - y0) / region.height]}


def split_grid(width: int, height: int, cols: int, rows: int,
               overlap_x: float, overlap_y: float) -> list[dict]:
    """均分图像并按局部图尺寸比例重叠；确保覆盖边界和全部像素。"""
    def intervals(length: int, count: int, overlap: float) -> list[tuple[int, int]]:
        if length <= 0 or not 1 <= count <= length:
            raise ValueError("等分数须为正整数且不能超过对应图片的像素数。")
        if not 0 <= overlap < 100:
            raise ValueError("重叠百分比须大于等于 0 且小于 100。")
        if count == 1:
            return [(0, length)]
        tile_size = math.ceil(length / (count - (count - 1) * overlap / 100))
        return [(round(i * (length - tile_size) / (count - 1)),
                 round(i * (length - tile_size) / (count - 1)) + tile_size)
                for i in range(count)]

    xs = intervals(width, cols, overlap_x)
    ys = intervals(height, rows, overlap_y)
    return [{"id": f"r{r:03d}_c{c:03d}", "row": r, "col": c, "source_bbox": [x0, y0, x1, y1]}
            for r, (y0, y1) in enumerate(ys, 1)
            for c, (x0, x1) in enumerate(xs, 1)]


def mark_nodes(image: Image.Image, nodes: list[dict], coordinate_key: str) -> None:
    draw = ImageDraw.Draw(image)
    radius = max(8, round(max(image.size) / 500))
    try:
        font = ImageFont.load_default(size=max(14, radius * 2))
    except TypeError:  # Pillow 10.0 does not support the size argument.
        font = ImageFont.load_default()
    for node in nodes:
        point = node[coordinate_key]
        if point is None:
            continue
        x, y = point
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="red", outline="white", width=max(1, radius // 5))
        draw.text((x + radius + 2, y - radius), node["id"].replace("node_", ""),
                  fill="red", font=font, stroke_width=max(1, radius // 8), stroke_fill="white")


def save_global_locations(image_path: Path, output: Path, nodes: list[dict]) -> None:
    """在高清图中绘制全局坐标。"""
    with Image.open(image_path) as img:
        global_image = img.convert("RGB")
    mark_nodes(global_image, nodes, "center_global_px")
    global_image.save(output / "global_marked.png")


def save_location_results(output: Path, region: dict, nodes: list[dict]) -> None:
    """保存局部标点及坐标。"""
    with Image.open(output / "region.png") as img:
        local = img.convert("RGB")
    mark_nodes(local, nodes, "center_local_px")
    local.save(output / "region_marked.png")
    save_json(output / "nodes.json", {"region": region, "nodes": nodes,
              "coordinate_system": "pixels_top_left",
              "note": "Model estimates are unverified; unresolved nodes are not drawn."})


def save_final_nodes(output: Path, nodes: list[dict]) -> None:
    """保存节点 ID、类型和高清图坐标；未定位节点单独保存。"""
    located = [{"id": node["id"], "type": node.get("type", "unknown"),
                "center_global_px": node["center_global_px"]}
               for node in nodes if node["center_global_px"] is not None]
    unresolved = [{"id": node["id"], "type": node.get("type", "unknown"), "center_global_px": None}
                  for node in nodes if node["center_global_px"] is None]
    save_json(output / "nodes.json", {"coordinate_system": "pixels_top_left", "nodes": located})
    save_json(output / "unresolved_candidates.json", {"coordinate_system": "pixels_top_left",
                                                        "nodes": unresolved})


def _starts(length: int, window: int, overlap: float) -> list[int]:
    if length <= window:
        return [0]
    step = max(1, int(window * (1 - overlap / 100)))
    starts = list(range(0, length - window + 1, step))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def mark_dedup_candidates(image: Image.Image, candidates: list[dict]) -> None:
    """为候选点绘制带引线的 ID，尽量避免近邻标签重叠。"""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    placed = []
    labels = []
    for candidate in candidates:
        x, y = candidate['center_local_px']
        alias = candidate['id']
        bounds = draw.textbbox((0, 0), alias, font=font)
        width, height = bounds[2] - bounds[0] + 8, bounds[3] - bounds[1] + 8
        positions = []
        for distance in (12, 36, 60, 90, 130):
            positions.extend([(x + distance, y - height / 2), (x - distance - width, y - height / 2),
                              (x - width / 2, y - distance - height), (x - width / 2, y + distance),
                              (x + distance, y - distance - height), (x - distance - width, y + distance)])
        choices = []
        for left, top in positions:
            left = max(0, min(image.width - width, left))
            top = max(0, min(image.height - height, top))
            box = (left, top, left + width, top + height)
            overlap = sum(max(0, min(box[2], other[2]) - max(box[0], other[0])) *
                          max(0, min(box[3], other[3]) - max(box[1], other[1])) for other in placed)
            choices.append((overlap, (left - x) ** 2 + (top - y) ** 2, box))
        box = min(choices)[2]
        placed.append(box)
        labels.append((x, y, alias, box))
    for x, y, _, box in labels:
        draw.line((x, y, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2), fill='red', width=2)
    for x, y, alias, box in labels:
        draw.rectangle(box, fill='white', outline='red', width=1)
        draw.text((box[0] + 4, box[1] + 1), alias, fill='red', font=font)
    # 先画所有白色外圈，再画所有红色内点，避免相邻点的白边盖住另一个点的中心。
    for x, y, _, _ in labels:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='white')
    for x, y, _, _ in labels:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill='red')


def recheck_pair_boxes(client: ModelClient, crop: Image.Image, crop_bbox: list[int],
                       local_points: list[list[float]], pair_dir: Path, resume: bool) -> dict:
    """分别识别两个标点所在的节点框；只用框体重叠度提供复核证据。"""
    boxes = []
    for index, point in enumerate(local_points, 1):
        marked = crop.copy()
        mark_target(marked, point)
        image_path = pair_dir / f'box_{index:03d}_marked.png'
        result_path = pair_dir / f'box_{index:03d}_response.json'
        marked.save(image_path)
        request = {'crop_global_px': crop_bbox, 'point_local_px': point,
                   'marked_image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest()}
        if resume and result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            if saved.get('request') != request:
                raise ValueError('节点框复核的截图或标点已改变，不能复用结果。')
            response = saved['response']
        else:
            prompt = (f'流程图局部截图宽 {crop.width} 像素、高 {crop.height} 像素。'
                      '红点标记目标节点，请只根据可见形状返回包住该节点的最小水平矩形框。'
                      '粗横条也可能是节点；不要读取节点文字或分析连线。'
                      '以截图左上角为原点，使用截图像素坐标返回 JSON 对象 '
                      '{"bbox":[左上x,左上y,右下x,右下y]}。无法判断或节点被截断返回 {"bbox":null}。')
            response = client.ask_json(prompt, [image_path], stage='dedup_box_review')
            save_json(result_path, {'request': request, 'response': response})
        boxes.append(local_to_global_box(response.get('bbox'), crop_bbox, point))
    if any(box is None for box in boxes):
        return {'bbox_global_px': boxes, 'iou': None}
    a, b = boxes
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
    return {'bbox_global_px': boxes, 'iou': intersection / union if union else None}


def _request_window_groups(client: ModelClient, image: Image.Image, path: Path, bbox: tuple[int, int, int, int],
                           nodes: list[dict], resume: bool) -> tuple[list, dict]:
    """在高清滑窗中标点，获取并缓存模型提出的重复分组。"""
    x0, y0, _, _ = bbox
    path.mkdir(exist_ok=True)
    crop = image.crop(bbox)
    crop.save(path / 'window.png')
    candidates = []
    for index, node in enumerate(nodes, 1):
        x, y = node['center_global_px']
        candidates.append({'id': f'D{index:03d}', 'node_id': node['id'],
                           'center_local_px': [x - x0, y - y0], 'type': node.get('type')})
    mark_dedup_candidates(crop, candidates)
    marked_path = path / 'marked.png'
    crop.save(marked_path)
    request = {'protocol': 'marked_image_v3', 'bbox_global_px': list(bbox),
               'marked_image_sha256': hashlib.sha256(marked_path.read_bytes()).hexdigest(),
               'candidates': candidates}
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    save_json(path / 'candidates.json', request)
    response_path = path / 'response.json'
    if resume and response_path.exists():
        saved = json.loads(response_path.read_text(encoding='utf-8'))
        if saved.get('request_hash') != request_hash:
            raise ValueError(f'{path.name} 的候选已改变，不能复用去重结果。')
        response = saved['response']
    else:
        prompt = ('这是一张流程图高清局部图，红点表示候选节点，每个红色 ID 通过红线指向对应候选点。'
                  '请仅根据图像判断哪些候选点标记的是同一个实际流程图节点。'
                  '只根据节点的可见形状判断，不要读取节点内部文字或分析节点之间的连线。'
                  '只有明确指向同一节点的 ID 才分组；不确定时不要合并。每个 ID 最多出现一次。'
                  '只返回 JSON 对象，例如 {"groups":[["D001","D002"]]}；没有重复返回 {"groups":[]}。'
                  '不要输出坐标，不要创造图中不存在的 ID。')
        print(f'去重窗口 {path.name}：检查 {len(nodes)} 个候选。', flush=True)
        response = client.ask_json(prompt, [marked_path], stage='dedup_window')
        save_json(response_path, {'request_hash': request_hash, 'response': response})
    groups = response.get('groups')
    if not isinstance(groups, list):
        raise ValueError(f'{path.name} 返回的去重分组无效。')
    return groups, {candidate['id']: candidate['node_id'] for candidate in candidates}


def _request_pair_check(client: ModelClient, image: Image.Image, first: dict, second: dict,
                        folder: Path, window_size: int, type_conflict: bool, resume: bool) -> dict:
    """单独复核相邻点；首次判断不可靠时比较两个节点框。"""
    first_id, second_id = first['id'], second['id']
    ax, ay = first['center_global_px']
    bx, by = second['center_global_px']
    width = min(window_size, 1200, image.width)
    height = min(window_size, 1200, image.height)
    left = max(0, min(image.width - width, round((ax + bx - width) / 2)))
    top = max(0, min(image.height - height, round((ay + by - height) / 2)))
    bbox = [left, top, left + width, top + height]
    original = image.crop(tuple(bbox))
    marked = original.copy()
    candidates = [{'id': 'D001', 'center_local_px': [ax - left, ay - top]},
                  {'id': 'D002', 'center_local_px': [bx - left, by - top]}]
    mark_dedup_candidates(marked, candidates)
    key = hashlib.sha256(json.dumps([first_id, second_id], ensure_ascii=False).encode()).hexdigest()[:16]
    path = folder / key
    path.mkdir(exist_ok=True)
    marked_path = path / 'marked.png'
    marked.save(marked_path)
    request = {'node_ids': [first_id, second_id], 'bbox_global_px': bbox,
               'marked_image_sha256': hashlib.sha256(marked_path.read_bytes()).hexdigest()}
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    save_json(path / 'candidates.json', request)
    response_path = path / 'response.json'
    if resume and response_path.exists():
        saved = json.loads(response_path.read_text(encoding='utf-8'))
        if saved.get('request_hash') != request_hash:
            raise ValueError('局部复核的图片或候选已改变，不能复用去重结果。')
        response = saved['response']
    else:
        prompt = ('这是一张流程图局部图。红色 D001 和 D002 分别通过红线指向一个候选点。'
                  '只观察流程图节点的可见形状，判断两个点是否标记了同一个节点。'
                  '不要读取节点文字、编号或分析连线。只在确定时返回 true 或 false；'
                  '不确定返回 null。只返回 JSON 对象：{"same_node":true}。')
        print(f'相邻候选复核：{first_id}、{second_id}。', flush=True)
        response = client.ask_json(prompt, [marked_path], stage='dedup_pair_review')
        save_json(response_path, {'request_hash': request_hash, 'response': response})
    same = response.get('same_node')
    if same is not None and not isinstance(same, bool):
        raise ValueError('相邻候选复核结果的 same_node 必须是 true、false 或 null。')
    check = {'node_ids': [first_id, second_id], 'same_node': same, 'path': str(path)}
    if same is not True or type_conflict:
        print(f'节点框重叠复核：{first_id}、{second_id}。', flush=True)
        check['box_check'] = recheck_pair_boxes(client, original, bbox,
                                                [candidate['center_local_px'] for candidate in candidates],
                                                path, resume)
    return check


def deduplicate_nodes(client: ModelClient, image_path: Path, output: Path, nodes: list[dict],
                      window_size: int = 2000, overlap: float = 25, resume: bool = False,
                      workers: int = 4) -> list[dict]:
    """高清滑窗去重，再复核相邻节点；保留原始候选坐标。"""
    if window_size <= 0 or not 0 <= overlap < 100:
        raise ValueError('无效的滑窗尺寸或重叠比例。')
    by_id = {node['id']: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError('候选节点 ID 不唯一。')
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    for node in nodes:
        point = node.get('center_global_px')
        if not isinstance(point, list) or len(point) != 2 or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in point) or not (0 <= point[0] < image.width and 0 <= point[1] < image.height):
            raise ValueError(f"候选节点 {node['id']} 缺少有效的高清图坐标。")

    parent = {node['id']: node['id'] for node in nodes}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def component_conflicts(members: list[str]) -> tuple[bool, bool, bool]:
        roots = {find(member) for member in members}
        combined = [node for key, node in by_id.items() if find(key) in roots]
        tiles = [node['tile_id'] for node in combined if node.get('tile_id')]
        types = {str(node.get('type')).strip().casefold() for node in combined
                 if node.get('type') is not None and str(node['type']).strip().casefold() not in ('', 'null', 'none', 'unknown')}
        too_far = any(math.dist(a['center_global_px'], b['center_global_px']) > 150
                      for i, a in enumerate(combined) for b in combined[i + 1:])
        return len(tiles) != len(set(tiles)), len(types) > 1, too_far

    evidence, rejected = [], []
    folder = output / 'dedup_windows'
    folder.mkdir(exist_ok=True)
    jobs = []
    for row, y0 in enumerate(_starts(image.height, window_size, overlap), 1):
        for col, x0 in enumerate(_starts(image.width, window_size, overlap), 1):
            x1, y1 = min(x0 + window_size, image.width), min(y0 + window_size, image.height)
            selected = [n for n in nodes if x0 <= n['center_global_px'][0] < x1 and y0 <= n['center_global_px'][1] < y1]
            if len(selected) >= 2:
                jobs.append((f'r{row:03d}_c{col:03d}', (x0, y0, x1, y1), selected))

    def request_groups(job):
        name, bbox, selected = job
        return _request_window_groups(client, image, folder / name, bbox, selected, resume)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        window_results = list(pool.map(request_groups, jobs))
    for job, (groups, aliases) in zip(jobs, window_results):
            name = job[0]
            counts = {}
            for group in groups:
                if isinstance(group, list):
                    for alias in set(a for a in group if isinstance(a, str)):
                        counts[alias] = counts.get(alias, 0) + 1
            for group in groups:
                if not isinstance(group, list) or len(group) < 2 or any(not isinstance(a, str) or a not in aliases for a in group) or len(set(group)) != len(group):
                    rejected.append({'window': name, 'group': group, 'reason': 'invalid_group'})
                    continue
                if any(counts[a] > 1 for a in group):
                    rejected.append({'window': name, 'group': group, 'reason': 'repeated_id_in_groups'})
                    continue
                members = [aliases[alias] for alias in group]
                same_tile, type_conflict, too_far = component_conflicts(members)
                if same_tile or type_conflict or too_far:
                    reason = 'same_tile_ambiguous' if same_tile else 'conflicting_type' if type_conflict else 'distance_too_large'
                    rejected.append({'window': name, 'group': group, 'node_ids': members, 'reason': reason})
                    continue
                evidence.append({'window': name, 'node_ids': members})
                root = find(members[0])
                for member in members[1:]:
                    parent[find(member)] = root

    focus_checks = []
    focus_dir = output / 'dedup_pair_checks'
    focus_dir.mkdir(exist_ok=True)
    for index, first in enumerate(nodes):
        for second in nodes[index + 1:]:
            first_id, second_id = first['id'], second['id']
            if find(first_id) == find(second_id) or math.dist(first['center_global_px'], second['center_global_px']) > 150:
                continue
            same_tile, type_conflict, too_far = component_conflicts([first_id, second_id])
            if same_tile or too_far:
                focus_checks.append({'node_ids': [first_id, second_id], 'same_node': None,
                                     'reason': 'same_tile_ambiguous' if same_tile else 'distance_too_large'})
                continue
            check = _request_pair_check(client, image, first, second, focus_dir, window_size, type_conflict, resume)
            focus_checks.append(check)
            same = check['same_node']
            box_iou = check.get('box_check', {}).get('iou')
            if (same is True and not type_conflict) or (box_iou is not None and box_iou >= 0.6):
                parent[find(second_id)] = find(first_id)
                method = 'focused_pair' if same is True and not type_conflict else 'focused_box_overlap'
                evidence.append({'window': check['path'], 'node_ids': [first_id, second_id], 'method': method})

    clusters = {}
    for node in nodes:
        clusters.setdefault(find(node['id']), []).append(node)
    result = [members[0] for members in clusters.values()]
    save_json(output / 'dedup_review.json', {'groups': evidence, 'rejected_groups': rejected,
              'focused_pair_checks': focus_checks,
              'note': '模型判断可能有误；冲突分组未合并，坐标保留原始候选。'})
    return result


def process_image_entities(image: Path, folder: Path, image_metadata: dict, args, client: Optional[ModelClient]) -> None:
    """原图切分、逐块抽取，再通过高清滑窗去重。"""
    if args.resume and (folder / "nodes.json").is_file() and (folder / "global_marked.png").is_file():
        print(f"{folder.name}：已完成，跳过。", flush=True)
        return
    tiles = split_grid(image_metadata["width"], image_metadata["height"], args.cols, args.rows,
                       args.overlap_x, args.overlap_y)
    tiles_dir = folder / "tiles"
    tiles_dir.mkdir(exist_ok=args.resume)
    with Image.open(image) as source_image:
        for tile in tiles:
            tile_dir = tiles_dir / tile["id"]
            tile_dir.mkdir(exist_ok=args.resume)
            tile_meta = tile_dir / "metadata.json"
            if args.resume and tile_meta.is_file() and (tile_dir / "region.png").is_file():
                saved = json.loads(tile_meta.read_text(encoding="utf-8"))
                if saved.get("source_bbox") != tile["source_bbox"]:
                    raise ValueError(f"{tile['id']} 的裁图范围与当前网格不一致。")
                tile.update(saved)
            else:
                if args.resume and (tile_dir / "nodes.json").exists():
                    raise ValueError(f"{tile['id']} 缺少裁图或元数据，不能复用已保存的节点。")
                tile.update(prepare_region(source_image, tuple(tile["source_bbox"]), tile_dir, args.tile_max_size))
                save_json(tile_meta, tile)
    save_json(folder / "tiles.json", {"cols": args.cols, "rows": args.rows,
              "overlap_x_percent": args.overlap_x, "overlap_y_percent": args.overlap_y,
              "tile_max_size": args.tile_max_size, "tiles": tiles})
    print(f"{folder.name}：生成 {len(tiles)} 张局部图。", flush=True)
    if args.split_only:
        return
    def locate_tile(tile):
        tile_dir = tiles_dir / tile["id"]
        tile_nodes = tile_dir / "nodes.json"
        located = None
        if args.resume and tile_nodes.is_file():
            saved = json.loads(tile_nodes.read_text(encoding="utf-8"))
            if saved.get("region", {}).get("source_bbox") != tile["source_bbox"] or not isinstance(saved.get("nodes"), list):
                raise ValueError(f"{tile['id']} 的缓存节点与局部图不一致。")
            located = saved["nodes"]
            print(f"{tile['id']}：复用 {len(located)} 个节点。", flush=True)
        if located is None:
            print(f"{tile['id']}：开始识别。", flush=True)
            located = extract_region_entities(client, tile_dir / "region.png", tile)
            save_location_results(tile_dir, tile, located)
            print(f"{tile['id']}：识别 {len(located)} 个节点。", flush=True)
        return located

    with ThreadPoolExecutor(max_workers=getattr(args, "workers", 4)) as pool:
        tile_results = list(pool.map(locate_tile, tiles))
    all_nodes = []
    for tile, located in zip(tiles, tile_results):
        all_nodes.extend([{**node, "id": f"{tile['id']}_{node['id']}", "tile_id": tile["id"]}
                          for node in located])
    save_json(folder / "nodes_before_dedup.json", {"nodes": all_nodes,
              "coordinate_system": "pixels_top_left", "note": "Original per-tile observations."})
    unresolved = [node for node in all_nodes if node["center_global_px"] is None]
    located_nodes = [node for node in all_nodes if node["center_global_px"] is not None]
    save_json(folder / "unresolved_candidates.json", {"coordinate_system": "pixels_top_left",
              "nodes": [{"id": node["id"], "type": node.get("type", "unknown"),
                         "center_global_px": None} for node in unresolved]})
    if unresolved:
        print(f"有 {len(unresolved)} 条候选未定位，单独保留，其余候选继续去重。", flush=True)
    deduplicated = deduplicate_nodes(client, image, folder, located_nodes,
                                     args.dedup_window, args.dedup_overlap, args.resume,
                                     getattr(args, "workers", 4))
    save_global_locations(image, folder, deduplicated)
    save_final_nodes(folder, deduplicated + unresolved)
    print(f"已定位候选去重：{len(located_nodes)} → {len(deduplicated)}；未定位：{len(unresolved)}。", flush=True)

