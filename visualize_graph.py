"""Draw node boxes and directed graph paths on the original high-resolution image."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def valid_point(point, width, height):
    return (isinstance(point, list) and len(point) == 2 and
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in point) and
            0 <= point[0] < width and 0 <= point[1] < height)


def put_label(draw, position, text, font, color, width, height):
    left, top = position
    bounds = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    w, h = bounds[2] - bounds[0] + 6, bounds[3] - bounds[1] + 6
    left = max(0, min(round(left), max(0, width-w)))
    top = max(0, min(round(top), max(0, height-h)))
    draw.rectangle((left, top, left+w, top+h), fill='white')
    draw.text((left+3, top+3-bounds[1]), text, font=font, fill=color, stroke_width=1, stroke_fill='white')


def draw_arrow(draw, path, color):
    # Put an arrow on the longest segment so it remains legible for short edges.
    segments = [(math.hypot(b[0]-a[0], b[1]-a[1]), a, b) for a, b in zip(path, path[1:])]
    length, a, b = max(segments, key=lambda segment: segment[0], default=(0, None, None))
    if length == 0:
        return None
    ux, uy = (b[0]-a[0])/length, (b[1]-a[1])/length
    cx, cy = a[0] + (b[0]-a[0])*0.55, a[1] + (b[1]-a[1])*0.55
    size = min(26, max(7, length*0.22))
    tip = (cx+ux*size/2, cy+uy*size/2)
    base = (cx-ux*size/2, cy-uy*size/2)
    normal = (-uy*size*0.45, ux*size*0.45)
    draw.polygon((tip, (base[0]+normal[0], base[1]+normal[1]),
                  (base[0]-normal[0], base[1]-normal[1])), fill=color)
    return cx, cy


def render(image, graph, evidence):
    if graph.get('coordinate_system') != 'pixels_top_left':
        raise ValueError('图 JSON 必须使用高清图左上角像素坐标。')
    width, height = image.size
    nodes = graph.get('nodes')
    edges = graph.get('edges')
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError('图 JSON 必须包含 nodes 和 edges 列表。')
    ids = [n.get('id') for n in nodes]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('节点 ID 缺失或重复。')
    edge_ids = [e.get('id') for e in edges]
    if any(not isinstance(i, str) or not i for i in edge_ids) or len(edge_ids) != len(set(edge_ids)):
        raise ValueError('边 ID 缺失或重复。')
    node_ids = set(ids)
    for node in nodes:
        box = node.get('bbox_global_px')
        if (not isinstance(box, list) or len(box) != 4 or
            not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box) or
            not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError(f"节点 {node['id']} 的框体无效。")
    for edge in edges:
        path = edge.get('path_global_px')
        if (edge.get('from') not in node_ids or edge.get('to') not in node_ids or
            not isinstance(path, list) or len(path) < 2 or not all(valid_point(p, width, height) for p in path)):
            raise ValueError(f"边 {edge['id']} 的端点或路径无效。")
    evidence_map = {e['edge_id']: e for e in evidence.get('edges', [])}
    if len(evidence_map) != len(evidence.get('edges', [])):
        raise ValueError('边证据包含重复 ID。')
    if evidence and set(evidence_map) != set(edge_ids):
        raise ValueError('边证据与图 JSON 的边 ID 不一致。')
    for edge in edges:
        if evidence and edge['path_global_px'] != evidence_map[edge['id']].get('path_global_px'):
            raise ValueError(f"边 {edge['id']} 的路径与证据不一致。")
    canvas = image.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    # Boxes and paths are global pixel coordinates; no image scaling is performed.
    for node in nodes:
        draw.rectangle(tuple(node['bbox_global_px']), outline='#eb3434', width=5)
    for edge in edges:
        path = [tuple(p) for p in edge['path_global_px']]
        info = evidence_map.get(edge['id'], {})
        color = '#ff9900' if info.get('via_gap') else '#00c4ec'
        draw.line(path, fill='black', width=9, joint='curve')
        draw.line(path, fill=color, width=5, joint='curve')
        center = draw_arrow(draw, path, color)
        if center:
            put_label(draw, (center[0]+8, center[1]+8), edge['id'], font, color, width, height)
    # Node labels use the full IDs used by graph.json.
    for node in nodes:
        x1, y1, _, _ = node['bbox_global_px']
        put_label(draw, (x1, y1-32), node['id'], font, '#c50000', width, height)
    return canvas


def render_box_comparison(image, audit):
    """Compare original and corrected boxes; never alter either JSON."""
    canvas = image.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    for node in audit.get('nodes', []):
        if node.get('status') != 'refined':
            continue
        original, corrected = node['original_bbox_global_px'], node['bbox_global_px']
        draw.rectangle(tuple(original), outline='#ef3030', width=6)
        draw.rectangle(tuple(corrected), outline='#00bb33', width=4)
        put_label(draw, (corrected[0], min(original[1], corrected[1])-32),
                  node['id'], font, '#117722', canvas.width, canvas.height)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--evidence', type=Path, help='输出图时标示经推测断口的边')
    parser.add_argument('--box-audit', type=Path, help='额外输出原始和校正框的对照图')
    parser.add_argument('--region', nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'), help='另存该全局范围内的局部图')
    parser.add_argument('--output', type=Path, required=True, help='新输出目录')
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding='utf-8'))
    evidence = json.loads(args.evidence.read_text(encoding='utf-8')) if args.evidence else {}
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    if args.region and not (0 <= args.region[0] < args.region[2] <= image.width and
                            0 <= args.region[1] < args.region[3] <= image.height):
        parser.error('--region 超出图片或坐标无效')
    marked = render(image, graph, evidence)
    args.output.mkdir(parents=True, exist_ok=False)
    marked.save(args.output/'global_marked.png')
    if args.box_audit:
        audit = json.loads(args.box_audit.read_text(encoding='utf-8'))
        render_box_comparison(image, audit).save(args.output/'box_comparison.png')
    if args.region:
        marked.crop(tuple(args.region)).save(args.output/'region_marked.png')
    print(f"完成：{len(graph['nodes'])} 个节点框，{len(graph['edges'])} 条边。")


if __name__ == '__main__':
    main()
