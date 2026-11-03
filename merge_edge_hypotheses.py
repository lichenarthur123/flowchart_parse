"""Combine edge hypotheses from several independent extraction profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def attachment_penalty(point, node):
    """Prefer connections near diamond vertices, not arbitrary nearby text."""
    if node.get('type') not in ('decision', 'merge'):
        return 0.0
    x, y = point['point_global_px']
    x1, y1, x2, y2 = node['bbox_global_px']
    side = min((abs(x-x1), 'left'), (abs(x-x2), 'right'), (abs(y-y1), 'top'), (abs(y-y2), 'bottom'))[1]
    if side in ('top', 'bottom'):
        return abs(x - (x1+x2)/2) / (x2-x1)
    return abs(y - (y1+y2)/2) / (y2-y1)


def same_route(first, second, tolerance=40):
    """Compare the physical polylines, independently of traversal direction."""
    def distances(path, other):
        result = []
        segments = list(zip(other, other[1:]))
        for start, end in zip(path, path[1:]):
            length = np.linalg.norm(np.subtract(end, start))
            for t in np.linspace(0, 1, max(2, int(length / 35) + 1)):
                point = np.asarray(start, dtype=float) * (1-t) + np.asarray(end, dtype=float) * t
                best = float('inf')
                for a, b in segments:
                    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                    delta = b-a
                    denom = np.dot(delta, delta)
                    ratio = np.clip(np.dot(point-a, delta) / denom, 0, 1) if denom else 0
                    best = min(best, float(np.linalg.norm(point - (a + ratio*delta))))
                result.append(best)
        return result
    if len(first) < 2 or len(second) < 2:
        return False
    # Different masks may stop a path a little early near a label. Compare
    # the bulk of the route, while keeping genuinely separate loops apart.
    return max(np.percentile(distances(first, second), 85),
               np.percentile(distances(second, first), 85)) <= tolerance


def merge(files, nodes):
    points, groups = [], {}
    region = None
    for index, file in enumerate(files, 1):
        data = json.loads(file.read_text(encoding='utf-8'))
        if data.get('coordinate_system') != 'pixels_top_left':
            raise ValueError(f'{file} 坐标系无效')
        if region is not None and region != data.get('region_global_px'):
            raise ValueError('所有候选文件必须使用相同的全局处理区域')
        region = data['region_global_px']
        mapping = {}
        for point in data['connection_points']:
            p = dict(point)
            p['id'] = f'{index}_{p["id"]}'
            mapping[point['id']] = p
            points.append(p)
        for trace in data['traces']:
            a = mapping[trace['start_point_id']]
            b = mapping[trace['end_point_id']]
            if len(a['possible_nodes']) != 1 or len(b['possible_nodes']) != 1:
                continue
            node_pair = tuple(sorted([a['possible_nodes'][0]['id'], b['possible_nodes'][0]['id']]))
            if node_pair[0] == node_pair[1]:
                continue
            score = sum(attachment_penalty(p, nodes[p['possible_nodes'][0]['id']]) for p in (a, b)) if nodes else 0.0
            alternative = {**trace, 'start_point_id': a['id'], 'end_point_id': b['id'], 'profile': str(file),
                           'attachment_penalty': round(score, 4)}
            groups.setdefault(node_pair, []).append(alternative)
    traces = []
    for pair, alternatives in sorted(groups.items()):
        alternatives.sort(key=lambda t: (t['attachment_penalty'], t.get('via_gap', False),
                                         len(t.get('via_junctions_global_px', [])), len(t['path_global_px'])))
        routes = []
        for alternative in alternatives:
            group = next((route for route in routes if same_route(route[0]['path_global_px'], alternative['path_global_px'])), None)
            if group is None:
                routes.append([alternative])
            else:
                group.append(alternative)
        for route in routes:
            selected = dict(route[0])
            selected.update(id=f'candidate_{len(traces)+1:03d}', node_ids=list(pair), status='reached_endpoint', alternatives=route)
            traces.append(selected)
    return {'coordinate_system': 'pixels_top_left', 'region_global_px': region,
            'note': '按节点对和路径几何分组的未确认假设；一对节点可有多条物理边。',
            'connection_points': points, 'traces': traces}


def arrow_evidence(gray, point, node, region):
    """A scale-dependent width heuristic, not an arrow-direction detector."""
    ox, oy = region[:2]
    gx, gy = point['point_global_px']
    x, y = gx - ox, gy - oy
    x1, y1, x2, y2 = node['bbox_global_px']
    side = min((abs(gx-x1), 'left'), (abs(gx-x2), 'right'),
               (abs(gy-y1), 'top'), (abs(gy-y2), 'bottom'))[1]
    vertical = side in ('top', 'bottom')
    edge = {'top': y1-oy, 'bottom': y2-oy, 'left': x1-ox, 'right': x2-ox}[side]
    pos = y if vertical else x
    low = min(pos, edge) - (15 if side in ('left', 'top') else 2)
    high = max(pos, edge) + (15 if side in ('right', 'bottom') else 2)
    widths = []
    for z in range(max(0, int(low)), min(gray.shape[0 if vertical else 1], int(high) + 1)):
        center = x if vertical else y
        limit = gray.shape[1 if vertical else 0]
        start, end = max(0, center-25), min(limit, center+26)
        strip = gray[z, start:end] if vertical else gray[start:end, z]
        black = strip < 150
        mid = center-start
        near_start, near_end = max(0, mid-2), min(len(black), mid+3)
        choices = np.flatnonzero(black[near_start:near_end])
        if not len(choices):
            widths.append(0)
            continue
        i = near_start + int(choices[0])
        a = b = i
        while a > 0 and black[a-1]:
            a -= 1
        while b+1 < len(black) and black[b+1]:
            b += 1
        widths.append(b-a+1)
    broad = sum(width >= 10 for width in widths)
    return {'side': side, 'max_width_px': max(widths, default=0),
            'wide_rows': broad, 'possible_arrow_toward_node': broad >= 5}


def flag_hypotheses(data, image, nodes, attachment_threshold):
    gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    points = {p['id']: p for p in data['connection_points']}
    for trace in data['traces']:
        endpoints = [points[trace[k]] for k in ('start_point_id', 'end_point_id')]
        trace['arrow_evidence'] = [arrow_evidence(gray, p, nodes[p['possible_nodes'][0]['id']], data['region_global_px']) for p in endpoints]
        # On very short straight connections, both search windows may see the
        # same arrowhead. Its width taper identifies which end it points to.
        coords = [p['point_global_px'] for p in endpoints]
        if all(e['possible_arrow_toward_node'] for e in trace['arrow_evidence']):
            dx, dy = abs(coords[0][0]-coords[1][0]), abs(coords[0][1]-coords[1][1])
            if max(dx, dy) <= 60 and min(dx, dy) <= 4:
                vertical = dy >= dx
                axis = 1 if vertical else 0
                cross = 0 if vertical else 1
                origin = data['region_global_px'][:2]
                boxes = [nodes[p['possible_nodes'][0]['id']]['bbox_global_px'] for p in endpoints]
                bounds = sorted([boxes[i][axis + (2 if coords[i][axis] < coords[1-i][axis] else 0)] for i in range(2)])
                center = round(sum(p[cross] for p in coords)/2) - origin[cross]
                widths = []
                for global_z in range(int(bounds[0]), int(bounds[1])+1):
                    z = global_z-origin[axis]
                    if z < 0 or z >= gray.shape[0 if vertical else 1]:
                        continue
                    limit = gray.shape[1 if vertical else 0]
                    lo, hi = max(0, center-25), min(limit, center+26)
                    strip = gray[z, lo:hi] if vertical else gray[lo:hi, z]
                    mid = center-lo
                    if not 0 <= mid < len(strip) or strip[mid] >= 150:
                        continue
                    a = b = mid
                    while a > 0 and strip[a-1] < 150: a -= 1
                    while b+1 < len(strip) and strip[b+1] < 150: b += 1
                    width = b-a+1
                    if width >= 10:
                        widths.append((global_z, width))
                if len(widths) >= 5:
                    slope = np.polyfit(*zip(*widths), 1)[0]
                    if abs(slope) >= 0.2:
                        target = max(range(2), key=lambda i: coords[i][axis]) if slope < 0 else min(range(2), key=lambda i: coords[i][axis])
                        for i, evidence in enumerate(trace['arrow_evidence']):
                            evidence['possible_arrow_toward_node'] = i == target
                            evidence['short_line_width_slope'] = round(float(slope), 3)
        flags = []
        if all(e['possible_arrow_toward_node'] for e in trace['arrow_evidence']):
            flags.append('possible_two_incoming_arrows')
        if any(p['possible_nodes'][0]['distance_px'] > attachment_threshold for p in endpoints):
            flags.append('weak_endpoint_attachment')
        trace['review_flags'] = flags
    return data


def draw(data, image_path, output):
    with Image.open(image_path) as image:
        result = image.crop(tuple(data['region_global_px'])).convert('RGB')
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    ox, oy = data['region_global_px'][:2]
    for trace in data['traces']:
        path = [(x-ox, y-oy) for x, y in trace['path_global_px']]
        color = 'orange' if trace.get('via_gap') else 'red' if trace.get('via_junctions_global_px') else 'lime'
        if len(path) > 1:
            draw.line(path, fill=color, width=4)
            x, y = path[len(path)//2]
            draw.text((x+5, y+5), trace['id'], fill=color, font=font, stroke_width=2, stroke_fill='black')
    result.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--nodes', type=Path, help='节点框体 JSON；用于选择更可能接入菱形顶点的路径')
    parser.add_argument('--attachment-threshold', type=float, default=32, help='超过该像素距离的端点标记待复核（默认 32）')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    nodes = {node['id']: node for node in json.loads(args.nodes.read_text(encoding='utf-8'))['nodes']} if args.nodes else {}
    data = merge(args.inputs, nodes)
    if nodes:
        with Image.open(args.image) as image:
            cropped = image.crop(tuple(data['region_global_px']))
            flag_hypotheses(data, cropped, nodes, args.attachment_threshold)
    (args.output/'merged_candidates.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if nodes:
        # Flags are experimental. Preserve flagged candidates in the full output.
        screened = {**data, 'note': '实验性筛选结果，未包含待复核候选；不是最终有向拓扑。',
                    'traces': [t for t in data['traces'] if not t['review_flags']]}
        (args.output/'screened_candidates.json').write_text(json.dumps(screened, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        flagged = {**data, 'note': '实验规则标记的待复核候选；不能直接视为错误。',
                   'traces': [t for t in data['traces'] if t['review_flags']]}
        (args.output/'flagged_candidates.json').write_text(json.dumps(flagged, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        draw(screened, args.image, args.output/'screened_marked.png')
        print(f"其中 {sum(bool(t['review_flags']) for t in data['traces'])} 个节点对被标记为待复核。")
    draw(data, args.image, args.output/'merged_marked.png')
    print(f'合并得到 {len(data["traces"])} 个节点对，全部为待复核假设。')


if __name__ == '__main__':
    main()
