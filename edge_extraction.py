"""Experimental, conservative flowchart connection tracing on a high-resolution image.

Coordinates in every output are global image pixels. Optional vision review
checks attachment at both ends, but does not infer arrow direction.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_nodes(path: Path, image_size: tuple[int, int]) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("coordinate_system") != "pixels_top_left" or not isinstance(data.get("nodes"), list):
        raise ValueError("节点 JSON 必须使用 pixels_top_left 坐标并包含 nodes 列表。")
    width, height = image_size
    seen, nodes = set(), []
    for node in data["nodes"]:
        node_id, box = node.get("id"), node.get("bbox_global_px")
        if not isinstance(node_id, str) or not node_id or node_id in seen:
            raise ValueError("节点 ID 缺失或重复。")
        if (not isinstance(box, list) or len(box) != 4 or
            any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in box)):
            raise ValueError(f"节点 {node_id} 的框体无效。")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"节点 {node_id} 的框体超出图片。")
        seen.add(node_id)
        nodes.append(node)
    return nodes


def prepare_line_image(image: Image.Image, nodes: list[dict], origin: tuple[int, int], min_line: int,
                       box_padding: int = 0, threshold: int | None = None) -> np.ndarray:
    """Preserve long horizontal/vertical strokes; mask node boxes before thinning."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mode = cv2.THRESH_BINARY_INV if threshold is not None else cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    _, binary = cv2.threshold(gray, threshold or 0, 255, mode)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_line, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line)))
    lines = cv2.bitwise_or(horizontal, vertical)
    ox, oy = origin
    height, width = lines.shape
    for node in nodes:
        x1, y1, x2, y2 = node["bbox_global_px"]
        # Clipping protects against rectangles outside the selected ROI.
        a, b = max(0, math.floor(x1 - ox - box_padding)), max(0, math.floor(y1 - oy - box_padding))
        c, d = min(width, math.ceil(x2 - ox + box_padding)), min(height, math.ceil(y2 - oy + box_padding))
        if a < c and b < d:
            lines[b:d, a:c] = 0
    return cv2.ximgproc.thinning(lines)


def image_threshold(image: Image.Image) -> int:
    """Compute one Otsu threshold from the complete image, shared by all tiles."""
    hist = np.asarray(image.convert('L').histogram(), dtype=np.float64)
    counts = np.cumsum(hist)
    weighted = np.cumsum(hist * np.arange(256))
    total = counts[-1]
    denominator = counts * (total - counts)
    between = np.zeros(256, dtype=np.float64)
    np.divide((weighted[-1] * counts - weighted * total) ** 2, denominator,
              out=between, where=denominator > 0)
    return int(np.argmax(between))


def prepare_tiled_skeleton(image: Image.Image, nodes: list[dict], origin: tuple[int, int],
                           min_line: int, box_padding: int, tile_size: int,
                           overlap: int, threshold: int | None = None) -> tuple[np.ndarray, list[dict]]:
    """Compute overlapping tiles, assembling disjoint cores into one pixel-aligned image."""
    if tile_size < 1 or overlap < min_line + box_padding:
        raise ValueError('切片重叠像素须不小于最短线段长度与节点屏蔽范围之和。')
    if threshold is None:
        threshold = image_threshold(image)
    width, height = image.size
    result = np.zeros((height, width), dtype=np.uint8)
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            x2, y2 = min(width, x + tile_size), min(height, y + tile_size)
            left, top = max(0, x-overlap), max(0, y-overlap)
            right, bottom = min(width, x2+overlap), min(height, y2+overlap)
            global_origin = (origin[0] + left, origin[1] + top)
            relevant = [n for n in nodes if n['bbox_global_px'][0] < origin[0]+right+box_padding and
                        n['bbox_global_px'][2] > origin[0]+left-box_padding and
                        n['bbox_global_px'][1] < origin[1]+bottom+box_padding and
                        n['bbox_global_px'][3] > origin[1]+top-box_padding]
            tile = image.crop((left, top, right, bottom))
            local = prepare_line_image(tile, relevant, global_origin, min_line, box_padding, threshold)
            result[y:y2, x:x2] = local[y-top:y2-top, x-left:x2-left]
            tiles.append({'id': f'tile_{len(tiles)+1:04d}',
                          'crop_global_px': [origin[0]+left, origin[1]+top, origin[0]+right, origin[1]+bottom],
                          'core_global_px': [origin[0]+x, origin[1]+y, origin[0]+x2, origin[1]+y2]})
    return result, tiles


def find_unmatched_seam_endpoints(skeleton: np.ndarray, origin: tuple[int, int], tile_size: int) -> list[dict]:
    """Record line endpoints touching a core boundary without a continuation."""
    height, width = skeleton.shape
    issues, seen = [], set()
    for axis, length in (('x', width), ('y', height)):
        for boundary in range(tile_size, length, tile_size):
            for offset in (-2, -1, 0, 1):
                coordinate = boundary + offset
                if not 0 <= coordinate < length:
                    continue
                line = skeleton[:, coordinate] if axis == 'x' else skeleton[coordinate, :]
                for v in np.flatnonzero(line).tolist():
                    point = (coordinate, v) if axis == 'x' else (v, coordinate)
                    if len(neighbors(point, skeleton)) > 1:
                        continue
                    key = (axis, boundary, point)
                    if key not in seen:
                        seen.add(key)
                        issues.append({'reason': 'unmatched_tile_boundary_endpoint',
                                       'axis': axis, 'boundary_global_px': boundary + origin[0 if axis == 'x' else 1],
                                       'point_global_px': [point[0]+origin[0], point[1]+origin[1]]})
    return issues


def distance_to_box(x: int, y: int, box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(max(x1 - x, 0, x - x2), max(y1 - y, 0, y - y2))


def points_outward(skeleton: np.ndarray, start: tuple[int, int], box: list[float], origin: tuple[int, int]) -> bool:
    """Reject short strokes and lines running parallel to the closest box side."""
    current = start
    visited = {current}
    for _ in range(16):
        options = [p for p in neighbors(current, skeleton) if p not in visited]
        if len(options) != 1:
            break
        current = options[0]
        visited.add(current)
    if len(visited) < 8:
        return False
    ox, oy = origin
    gx, gy = start[0] + ox, start[1] + oy
    x1, y1, x2, y2 = box
    # Boxes use a right/bottom exclusive mask; measure from the same boundary.
    nearest_x = min(max(gx, x1), x2 - 1)
    nearest_y = min(max(gy, y1), y2 - 1)
    away_x, away_y = gx - nearest_x, gy - nearest_y
    line_x, line_y = current[0] - start[0], current[1] - start[1]
    lengths = math.hypot(away_x, away_y) * math.hypot(line_x, line_y)
    return bool(lengths and (away_x * line_x + away_y * line_y) / lengths >= 0.5)


def find_connection_points(skeleton: np.ndarray, nodes: list[dict], origin: tuple[int, int], max_gap: int) -> list[dict]:
    """Find skeleton endpoints near a box; proximity is not proof of connection."""
    foreground = (skeleton > 0).astype(np.uint8)
    degree = pixel_degree(foreground)
    ys, xs = np.where((foreground == 1) & (degree <= 1))
    ox, oy = origin
    points = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        gx, gy = x + ox, y + oy
        near = sorted((distance_to_box(gx, gy, node["bbox_global_px"]), node["id"], node["bbox_global_px"]) for node in nodes)
        near = [(round(distance, 2), node_id) for distance, node_id, box in near
                if distance <= max_gap and points_outward(skeleton, (x, y), box, origin)]
        if not near:
            continue
        # Keep all nearby nodes, so an ambiguous attachment is never silently guessed.
        points.append({"id": f"p{len(points) + 1:04d}", "point_global_px": [gx, gy],
                       "possible_nodes": [{"id": n, "distance_px": d} for d, n in near]})
    return points


def pixel_degree(foreground: np.ndarray) -> np.ndarray:
    """Count 8-neighbors, excluding diagonals when a cardinal route exists."""
    fg = np.pad(foreground > 0, 1)
    center = fg[1:-1, 1:-1]
    up, down, left, right = fg[:-2, 1:-1], fg[2:, 1:-1], fg[1:-1, :-2], fg[1:-1, 2:]
    degree = sum(a.astype(np.uint8) for a in (up, down, left, right))
    for diagonal, a, b in ((fg[:-2, :-2], up, left), (fg[:-2, 2:], up, right),
                           (fg[2:, :-2], down, left), (fg[2:, 2:], down, right)):
        degree += (diagonal & ~a & ~b).astype(np.uint8)
    return degree * center


def neighbors(point: tuple[int, int], skeleton: np.ndarray) -> list[tuple[int, int]]:
    x, y = point
    h, w = skeleton.shape
    return [(xx, yy) for yy in range(max(0, y - 1), min(h, y + 2))
            for xx in range(max(0, x - 1), min(w, x + 2))
            if (xx, yy) != point and skeleton[yy, xx] and
            (xx == x or yy == y or not (skeleton[y, xx] or skeleton[yy, x]))]


def prune_short_spurs(skeleton: np.ndarray, nodes: list[dict], origin: tuple[int, int],
                      max_length: int, protect_gap: int) -> tuple[np.ndarray, list[dict]]:
    """Remove short dead-end offshoots, while preserving node and region endpoints.

    Save removed pixels for diagnostics: a broken real line may look like a spur.
    No gaps or crossings are joined by this operation.
    """
    result = skeleton.copy()
    removed = []
    height, width = result.shape
    if max_length <= 0:
        return result, removed
    for _ in range(3):
        fg = (result > 0).astype(np.uint8)
        degree = pixel_degree(fg)
        ys, xs = np.where((fg == 1) & (degree == 1))
        paths = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            gx, gy = x + origin[0], y + origin[1]
            if (x <= max_length or y <= max_length or x >= width - 1 - max_length or
                    y >= height - 1 - max_length or
                    any(distance_to_box(gx, gy, n["bbox_global_px"]) <= protect_gap and
                        points_outward(result, (x, y), n["bbox_global_px"], origin) for n in nodes)):
                continue
            path = [(x, y)]
            previous = None
            current = (x, y)
            reached_junction = False
            while len(path) <= max_length:
                options = [p for p in neighbors(current, result) if p != previous]
                if len(options) > 1:
                    reached_junction = True
                    break
                if not options:
                    break
                previous, current = current, options[0]
                path.append(current)
            if reached_junction and len(path) <= max_length:
                # Keep the junction pixel itself; remove only the dead-end branch.
                paths.append(path[:-1])
        if not paths:
            break
        for path in paths:
            if not path:
                continue
            for x, y in path:
                result[y, x] = 0
            removed.append({"endpoint_global_px": [path[0][0] + origin[0], path[0][1] + origin[1]],
                            "length_px": len(path)})
    return result, removed


def bridge_gaps(skeleton: np.ndarray, nodes: list[dict], origin: tuple[int, int],
                max_gap: int) -> tuple[np.ndarray, list[dict], set]:
    """Propose straight bridges across interruptions, including to a line body."""
    result = skeleton.copy()
    if max_gap <= 0:
        return result, [], set()
    fg = (result > 0).astype(np.uint8)
    degree = pixel_degree(fg)
    ys, xs = np.where((fg == 1) & (degree == 1))
    ends = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        gx, gy = x + origin[0], y + origin[1]
        # A label can interrupt a branch immediately outside a large diamond.
        # Excluding the entire node_gap neighbourhood loses that branch; the
        # bridge itself is still rejected below if it crosses any node box.
        if any(distance_to_box(gx, gy, n['bbox_global_px']) == 0 for n in nodes):
            continue
        current = (x, y)
        seen = {current}
        for _ in range(25):
            options = [p for p in neighbors(current, result) if p not in seen]
            if len(options) != 1:
                break
            current = options[0]
            seen.add(current)
        dx, dy = current[0] - x, current[1] - y
        if len(seen) >= 20 and (abs(dx) >= 15 or abs(dy) >= 15):
            ends.append((x, y, dx, dy))
    # Group by exact row/column, avoiding an all-pairs scan over the full page.
    rows, columns = {}, {}
    for i, (x, y, dx, dy) in enumerate(ends):
        if abs(dx) >= 15:
            rows.setdefault(y, []).append((x, i, dx))
        if abs(dy) >= 15:
            columns.setdefault(x, []).append((y, i, dy))
    pairs = set()
    for groups in (rows, columns):
        for group in groups.values():
            group.sort()
            for k, (pos, i, direction) in enumerate(group):
                for other_pos, j, other_direction in group[k+1:]:
                    distance = other_pos - pos
                    if distance > max_gap:
                        break
                    if distance > 2 and direction < 0 and other_direction > 0:
                        pairs.add((distance, min(i, j), max(i, j)))
    # Nearest mutual matches only; never bridge through a node box.
    nearest = {}
    for distance, i, j in sorted(pairs):
        nearest.setdefault(i, j)
        nearest.setdefault(j, i)
    bridges, bridge_pixels = [], set()
    for distance, i, j in sorted(pairs):
        if nearest.get(i) != j or nearest.get(j) != i:
            continue
        x1, y1, _, _ = ends[i]
        x2, y2, _, _ = ends[j]
        points = ([(x, y1) for x in range(min(x1, x2), max(x1, x2) + 1)] if y1 == y2 else
                  [(x1, y) for y in range(min(y1, y2), max(y1, y2) + 1)])
        if any(any(n['bbox_global_px'][0] <= x + origin[0] <= n['bbox_global_px'][2] and
                       n['bbox_global_px'][1] <= y + origin[1] <= n['bbox_global_px'][3] for n in nodes)
               for x, y in points):
            continue
        for x, y in points:
            if not result[y, x]:
                bridge_pixels.add((x, y))
                result[y, x] = 255
        bridges.append({'from_global_px': [x1 + origin[0], y1 + origin[1]],
                        'to_global_px': [x2 + origin[0], y2 + origin[1]], 'gap_px': distance})
    # Text can hide a branch between an endpoint and an existing corner or
    # junction. Endpoint-only matching misses this because the other end has
    # multiple neighbours. Require a long continuation beyond the target;
    # every bridge is still recorded as a hypothesis, not a confirmed line.
    h, w = fg.shape
    for x, y, dx, dy in ends:
        horizontal = abs(dx) >= 15 and abs(dx) >= abs(dy)
        vertical = abs(dy) >= 15 and abs(dy) > abs(dx)
        if not (horizontal or vertical):
            continue
        sign = -1 if (dx if horizontal else dy) > 0 else 1
        for distance in range(3, max_gap + 1):
            tx = x + sign*distance if horizontal else x
            ty = y if horizontal else y + sign*distance
            if not (1 <= tx < w-1 and 1 <= ty < h-1):
                break
            if not fg[ty, tx]:
                continue
            if degree[ty, tx] < 2:
                break  # Endpoint-to-endpoint bridges are handled above.
            support = 0
            for step in range(1, 21):
                sx = tx + sign*step if horizontal else tx
                sy = ty if horizontal else ty + sign*step
                if not (1 <= sx < w-1 and 1 <= sy < h-1):
                    break
                if any(fg[sy+offset, sx] if horizontal else fg[sy, sx+offset] for offset in (-1, 0, 1)):
                    support += 1
            if support < 15:
                break
            points = ([(px, y) for px in range(min(x, tx), max(x, tx)+1)] if horizontal else
                      [(x, py) for py in range(min(y, ty), max(y, ty)+1)])
            if any(any(n['bbox_global_px'][0] <= px+origin[0] <= n['bbox_global_px'][2] and
                       n['bbox_global_px'][1] <= py+origin[1] <= n['bbox_global_px'][3] for n in nodes)
                   for px, py in points):
                break
            added = False
            for px, py in points:
                if not result[py, px]:
                    result[py, px] = 255
                    bridge_pixels.add((px, py))
                    added = True
            if added:
                bridges.append({'from_global_px': [x+origin[0], y+origin[1]],
                                'to_global_px': [tx+origin[0], ty+origin[1]],
                                'gap_px': distance, 'kind': 'endpoint_to_line'})
            break
    return result, bridges, bridge_pixels


def trace_line(start: dict, skeleton: np.ndarray, endpoint_map: dict, origin: tuple[int, int],
               bridge_pixels: set | None = None) -> dict:
    """Walk an 8-connected skeleton, stopping at ambiguities; never infer direction."""
    ox, oy = origin
    gx, gy = start["point_global_px"]
    current = (gx - ox, gy - oy)
    visited = {current}
    path = [current]
    while True:
        options = [p for p in neighbors(current, skeleton) if p not in visited]
        if len(options) > 1:
            status, reason = "ambiguous", "junction"
            branch_options = []
            for i, option in enumerate(options):
                # Extend each proposed branch just far enough to annotate the crop.
                branch = [current, option]
                branch_seen = visited | {option}
                branch_current = option
                for _ in range(50):
                    next_points = [p for p in neighbors(branch_current, skeleton) if p not in branch_seen]
                    if len(next_points) != 1:
                        break
                    branch_current = next_points[0]
                    branch_seen.add(branch_current)
                    branch.append(branch_current)
                branch_options.append({"id": chr(ord("A") + i),
                                       "path_global_px": [[x + ox, y + oy] for x, y in branch]})
            break
        if not options:
            target = endpoint_map.get(current)
            status = "reached_endpoint" if target and target["id"] != start["id"] else "unresolved"
            reason = None if status == "reached_endpoint" else "line_end_or_loop"
            break
        # At a corner prefer a cardinal step; visited pixels prevent returning along a path.
        options.sort(key=lambda p: abs(p[0] - current[0]) + abs(p[1] - current[1]))
        current = options[0]
        if current in visited:
            status, reason = "unresolved", "loop"
            break
        visited.add(current)
        path.append(current)
        if len(path) > skeleton.size:
            status, reason = "unresolved", "trace_limit"
            break
    points = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(points, 1.5, False).reshape(-1, 2).tolist() if len(path) > 1 else [list(current)]
    result = {"start_point_id": start["id"], "status": status,
              "path_global_px": [[x + ox, y + oy] for x, y in simplified],
              "via_gap": bool(bridge_pixels and not visited.isdisjoint(bridge_pixels))}
    if status == "reached_endpoint":
        result["end_point_id"] = target["id"]
    else:
        result.update(reason=reason, position_global_px=[current[0] + ox, current[1] + oy])
        if status == "ambiguous":
            result["branch_options"] = branch_options
    return result


def explore_branches(start: dict, skeleton: np.ndarray, endpoint_map: dict,
                     origin: tuple[int, int], max_junctions: int = 2,
                     max_states: int = 64, bridge_pixels: set | None = None,
                     unresolved: list[dict] | None = None) -> list[dict]:
    """Enumerate possible routes through junctions, never assuming they connect.

    Every resulting path is a hypothesis; later vision/human review must decide
    if the path really connects at each junction.
    """
    ox, oy = origin
    sx, sy = start["point_global_px"]
    first = (sx - ox, sy - oy)
    stack = [(first, {first}, [first], [])]
    results = []
    states = 0
    path_limit = max(30000, 2 * sum(skeleton.shape))
    def record(reason, point):
        if unresolved is not None:
            unresolved.append({'start_point_id': start['id'], 'reason': reason,
                               'position_global_px': [point[0] + ox, point[1] + oy]})
    while stack and states < max_states:
        current, visited, path, junctions = stack.pop()
        states += 1
        while len(path) <= path_limit:
            endpoint = endpoint_map.get(current)
            if endpoint and endpoint['id'] != start['id']:
                source_ids = {n['id'] for n in start['possible_nodes']}
                target_ids = {n['id'] for n in endpoint['possible_nodes']}
                if source_ids.isdisjoint(target_ids):
                    simplified = cv2.approxPolyDP(np.array(path, dtype=np.int32).reshape(-1, 1, 2), 1.5, False)
                    results.append({'start_point_id': start['id'], 'end_point_id': endpoint['id'],
                                    'status': 'reached_endpoint', 'via_junctions_global_px': junctions,
                                    'via_gap': bool(bridge_pixels and not visited.isdisjoint(bridge_pixels)),
                                    'path_global_px': [[x + ox, y + oy] for x, y in simplified.reshape(-1, 2).tolist()]})
                    break
            options = [p for p in neighbors(current, skeleton) if p not in visited]
            if not options:
                record('line_end_or_loop', current)
                break
            if len(options) > 1:
                if len(junctions) >= max_junctions:
                    record('junction_limit', current)
                    break
                new_junctions = junctions + [[current[0] + ox, current[1] + oy]]
                for option in options:
                    stack.append((option, visited | {option}, path + [option], new_junctions))
                break
            current = options[0]
            visited.add(current)
            path.append(current)
        else:
            record('path_limit', current)
    if stack:
        record('search_state_limit', stack[-1][0])
    return results


def branch_hypotheses(skeleton: np.ndarray, points: list[dict], region: list[int],
                      bridges: list[dict], bridge_pixels: set, max_junctions: int,
                      max_states: int) -> dict:
    origin = tuple(region[:2])
    endpoint_map = {(p['point_global_px'][0] - origin[0], p['point_global_px'][1] - origin[1]): p for p in points}
    hypotheses, seen_paths, unresolved = [], set(), []
    for point in points:
        for trace in explore_branches(point, skeleton, endpoint_map, origin,
                                      max_junctions=max_junctions, max_states=max_states,
                                      bridge_pixels=bridge_pixels, unresolved=unresolved):
            pair = tuple(sorted((trace['start_point_id'], trace['end_point_id'])))
            geometry = tuple(sorted(tuple(p) for p in trace['via_junctions_global_px']))
            path = tuple(tuple(p) for p in trace['path_global_px'])
            key = (pair, geometry, min(path, tuple(reversed(path))))
            if key not in seen_paths:
                seen_paths.add(key)
                hypotheses.append(trace)
    return {'coordinate_system': 'pixels_top_left', 'region_global_px': region,
            'note': '分叉处的候选路径不保证交叉处连通、箭头方向或节点归属。',
            'gap_hypotheses': bridges, 'connection_points': points, 'traces': hypotheses,
            'unresolved_traces': unresolved}


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def review_crop(image: Image.Image, region: list[int], point: dict, node: dict, path: Path) -> None:
    """Show the real image and a small circle at the program-calculated line endpoint."""
    ox, oy = region[:2]
    gx, gy = point["point_global_px"]
    left = max(0, gx - ox - 450)
    top = max(0, gy - oy - 450)
    right = min(image.width, gx - ox + 450)
    bottom = min(image.height, gy - oy + 450)
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    x, y = gx - ox - left, gy - oy - top
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline="red", width=3)
    x1, y1, x2, y2 = node["bbox_global_px"]
    draw.rectangle((x1 - ox - left - 8, y1 - oy - top - 8,
                    x2 - ox - left + 8, y2 - oy - top + 8), outline="blue", width=2)
    # A small outline avoids covering the line and arrowhead.
    crop.save(path)


def review_paths(client, image: Image.Image, region: list[int], nodes: list[dict],
                 points: list[dict], traces: list[dict], output: Path) -> dict:
    """Ask the model about end attachments. Unreviewed or unclear remains unresolved."""
    node_map = {n["id"]: n for n in nodes}
    point_map = {p["id"]: p for p in points}
    folder = output / "vision_review"
    folder.mkdir()
    reviewed = []
    calls = 0
    for trace in traces:
        if trace["status"] != "reached_endpoint":
            continue
        ends = [point_map[trace["start_point_id"]], point_map[trace["end_point_id"]]]
        item = {"id": f"edge_{len(reviewed)+1:04d}", "point_ids": [p["id"] for p in ends], "path_global_px": trace["path_global_px"],
                "node_ids": [p["possible_nodes"][0]["id"] for p in ends], "direction": "unknown",
                "pruning_affected": trace.get("pruning_affected", False), "via_gap": trace.get("via_gap", False)}
        if any(len(p["possible_nodes"]) != 1 for p in ends) or item["node_ids"][0] == item["node_ids"][1]:
            item.update(status="unresolved", reason="节点归属不唯一或两端为同一节点")
            reviewed.append(item)
            continue
        if any(not (region[0] <= node_map[node_id]["bbox_global_px"][0] and
                        region[1] <= node_map[node_id]["bbox_global_px"][1] and
                        node_map[node_id]["bbox_global_px"][2] <= region[2] and
                        node_map[node_id]["bbox_global_px"][3] <= region[3]) for node_id in item["node_ids"]):
            item.update(status="unresolved", reason="节点被处理区域截断")
            reviewed.append(item)
            continue
        paths = []
        for index, (point, node_id) in enumerate(zip(ends, item["node_ids"]), 1):
            path = folder / f"{len(reviewed)+1:04d}_end{index}.png"
            review_crop(image, region, point, node_map[node_id], path)
            paths.append(path)
        prompt = (
            "两张图片分别是同一条候选线的两端。每张图片中的红色细圆圈仅标记程序找到的线端点，"
            "蓝框标记目标节点范围，红圈和蓝框不是流程图原始内容。请分别判断："
            "红圈处的流程连线是否确实直接连接到蓝框内的目标节点本体。"
            "注意文字笔画、节点边框、经过但没有连接的线都不是有效连接；箭头连接到节点也算有效。"
            "如果圈住的不是线、看不清节点或无法确认，返回 uncertain。不要根据业务含义猜测。"
            "只返回 JSON：{\"first\":\"yes/no/uncertain\",\"second\":\"yes/no/uncertain\"}。"
        )
        try:
            calls += 1
            response = client.ask_json(prompt, paths, stage='edge_vision_review')
            first, second = response.get("first"), response.get("second")
            if first not in ("yes", "no", "uncertain") or second not in ("yes", "no", "uncertain"):
                first = second = "uncertain"
            status = "visually_supported" if first == second == "yes" else "rejected" if "no" in (first, second) else "unresolved"
            item.update(status=status, endpoint_review=[first, second])
        except Exception as exc:
            # Never persist provider response or secrets. Keep failures available for review.
            item.update(status="unresolved", reason=f"模型调用失败：{type(exc).__name__}")
        reviewed.append(item)
        save_json(output / "reviewed_edges.json", {"model_calls": calls, "edges": reviewed})
        print(f"视觉复核 {calls}：{item['status']}", flush=True)
    return {"coordinate_system": "pixels_top_left", "model_calls": calls, "edges": reviewed}


def review_junctions(client, image: Image.Image, region: list[int], traces: list[dict], output: Path) -> dict:
    """Record model suggestions for ambiguous branches without inventing final paths."""
    folder = output / "junction_review"
    folder.mkdir()
    results = []
    calls = 0
    colors = ["#24c44a", "#0089ff", "#ffad00", "#cb41f5", "#00bbbb", "#e366a6", "#eeee00", "#aaaaaa"]
    try:
        font = ImageFont.load_default(size=30)
    except TypeError:
        font = ImageFont.load_default()
    for trace in traces:
        if trace["status"] != "ambiguous":
            continue
        gx, gy = trace["position_global_px"]
        left, top = max(0, gx - region[0] - 400), max(0, gy - region[1] - 400)
        right, bottom = min(image.width, gx - region[0] + 400), min(image.height, gy - region[1] + 400)
        crop = image.crop((left, top, right, bottom))
        draw = ImageDraw.Draw(crop)
        def local(path):
            return [(x - region[0] - left, y - region[1] - top) for x, y in path]
        incoming = local(trace["path_global_px"])
        if len(incoming) > 1:
            draw.line(incoming, fill="red", width=4)
        for option, color in zip(trace["branch_options"], colors):
            branch = local(option["path_global_px"])
            if len(branch) > 1:
                draw.line(branch, fill=color, width=4)
            x, y = branch[-1]
            draw.text((x + 7, y + 7), option["id"], fill=color, font=font, stroke_width=2, stroke_fill="black")
        path = folder / f"{trace['start_point_id']}.png"
        crop.save(path)
        item = {"start_point_id": trace["start_point_id"], "position_global_px": trace["position_global_px"],
                "image": str(path.relative_to(output)), "options": [o["id"] for o in trace["branch_options"]]}
        # If short branches collapse into the same pixels, labels are unreadable
        # and the VLM cannot reliably distinguish them. Preserve for review.
        if any(len(option["path_global_px"]) < 20 for option in trace["branch_options"]):
            item.update(choices=[], uncertain=True, reason="候选分支过短，无法清晰标注")
            results.append(item)
            save_json(output / "junction_reviews.json", {"model_calls": calls, "reviews": results})
            print(f"分叉复核 {len(results)}：标注不清，跳过模型调用", flush=True)
            continue
        prompt = (
            "图片是流程图高清局部。程序用红线标记已经追踪到交叉处的路线，用不同颜色及字母标记"
            "从该处出发的候选后续线段；这些彩色标记不是原图内容。只根据原图可见的连接判断："
            "红线是否与候选线段直接相连？可能有多个分支；交叉穿过并不一定相连。"
            "图像看不清或无法区分交叉和连接时选 uncertain，不要猜测。"
            "只返回 JSON，例如 {\"choices\":[\"A\",\"B\"],\"uncertain\":false}，"
            "无法确定返回 {\"choices\":[],\"uncertain\":true}。"
        )
        try:
            calls += 1
            response = client.ask_json(prompt, [path], stage='edge_junction_review')
            choices, uncertain = response.get("choices"), response.get("uncertain")
            if (not isinstance(choices, list) or any(not isinstance(c, str) or c not in item["options"] for c in choices)
                    or not isinstance(uncertain, bool) or (not uncertain and not choices)):
                choices, uncertain = [], True
            item.update(choices=choices if not uncertain else [], uncertain=uncertain)
        except Exception as exc:
            item.update(choices=[], uncertain=True, reason=f"模型调用失败：{type(exc).__name__}")
        results.append(item)
        save_json(output / "junction_reviews.json", {"model_calls": calls, "reviews": results})
        print(f"分叉复核 {len(results)}：{'待确定' if item['uncertain'] else ','.join(item['choices'])}", flush=True)
    return {"coordinate_system": "pixels_top_left", "model_calls": calls, "reviews": results}


def draw_reviewed_paths(image: Image.Image, region: list[int], review: dict, output: Path) -> None:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    for edge in review["edges"]:
        points = [(x - region[0], y - region[1]) for x, y in edge["path_global_px"]]
        if len(points) < 2:
            continue
        color = {"visually_supported": "lime", "rejected": "red"}.get(edge["status"], "orange")
        if edge.get("via_gap") or edge.get("pruning_affected"):
            color = "orange"
        draw.line(points, fill=color, width=5)
        x, y = points[len(points) // 2]
        draw.text((x + 8, y + 8), edge["id"], fill=color, font=font, stroke_width=2, stroke_fill="black")
    result.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="实验：查找流程图节点候选连接点并追踪线条")
    parser.add_argument("image", type=Path)
    parser.add_argument("nodes", type=Path, help="节点框体 JSON，例如 boxes.json")
    parser.add_argument("--output", type=Path, required=True, help="新建输出目录")
    parser.add_argument("--region", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="全局坐标下的处理区域")
    parser.add_argument("--tile-size", type=int, default=2000, help="切片核心区域的最大宽高；0 关闭分块（默认 2000）")
    parser.add_argument("--tile-overlap", type=int, default=128, help="每个核心区域四周的重叠上下文像素（默认 128）")
    parser.add_argument("--max-gap", type=int, default=25, help="候选线头与框体最大距离（像素，默认 25）")
    parser.add_argument("--min-line", type=int, default=24, help="提取水平/垂直线段的最小长度（像素，默认 24）")
    parser.add_argument("--box-padding", type=int, default=12, help="节点框向外屏蔽像素（默认 12）")
    parser.add_argument("--spur-length", type=int, default=16, help="移除的短末端毛刺最大长度，0 表示关闭（默认 16）")
    parser.add_argument("--max-line-gap", type=int, default=0, help="推测连接对齐断口的最大像素距离；0 关闭（默认 0）")
    parser.add_argument("--vision-review", action="store_true", help="使用已配置的多模态模型复核候选路径的两端")
    parser.add_argument("--junction-review", action="store_true", help="使用多模态模型复核分叉（仅输出建议，不自动生成边）")
    parser.add_argument("--explore-branches", action="store_true", help="枚举分叉处可能路径，结果均须复核")
    parser.add_argument("--max-junctions", type=int, default=8, help="每条候选路径最多经过的分叉数（默认 8）")
    parser.add_argument("--max-branch-states", type=int, default=512, help="每个起点的最多搜索状态数（默认 512）")
    parser.add_argument("--candidates-only", action="store_true", help="只生成调试候选，不构建节点和边 JSON")
    parser.add_argument("--no-box-refinement", action="store_true", help="跳过基于图像轮廓的保守框体校正")
    parser.add_argument("--no-ocr", action="store_true", help="跳过边文字识别")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="模型配置文件")
    parser.add_argument("--unresolved-nodes", type=Path, action="append", default=[],
                        help="实体阶段未确认节点 JSON；可多次指定")
    args = parser.parse_args()
    if (args.max_gap < 0 or args.min_line < 2 or args.box_padding < 0 or args.spur_length < 0 or
            args.max_line_gap < 0 or args.max_junctions < 0 or args.max_branch_states < 1):
        parser.error("距离参数必须非负，--min-line 必须至少为 2。")
    required_overlap = args.min_line + max(args.box_padding, 24 if not args.candidates_only else 0)
    if args.tile_size < 0 or args.tile_overlap < 0 or (args.tile_size and args.tile_overlap < required_overlap):
        parser.error(f"切片尺寸必须非负，切片上下文至少为 {required_overlap} 像素。")
    with Image.open(args.image) as im:
        size = im.size
        region = args.region or [0, 0, *size]
        x1, y1, x2, y2 = region
        if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            parser.error("--region 超出图片或坐标无效。")
        image = im.crop(tuple(region)).convert("RGB")
    nodes = load_nodes(args.nodes, size)
    entity_data = json.loads(args.nodes.read_text(encoding='utf-8'))
    if not args.candidates_only and entity_data.get('edges', []) != []:
        parser.error('输入实体 JSON 已包含非空 edges，拒绝覆盖。')
    unresolved_paths = list(args.unresolved_nodes)
    adjacent_unresolved = args.nodes.parent / 'unresolved_boxes.json'
    if adjacent_unresolved.is_file() and adjacent_unresolved.resolve() not in {p.resolve() for p in unresolved_paths}:
        unresolved_paths.append(adjacent_unresolved)
    unresolved_nodes = []
    for path in unresolved_paths:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value.get('nodes'), list):
            parser.error(f'未确认节点文件缺少 nodes 列表：{path}')
        unresolved_nodes.extend({'source': str(path), 'node': node} for node in value['nodes'])
    args.output.mkdir(parents=True, exist_ok=False)
    box_audit = []
    if not args.no_box_refinement:
        from node_box_refinement import refine_node_boxes
        nodes, box_audit = refine_node_boxes(image, nodes, (x1, y1))
        entity_data = {**entity_data, 'nodes': nodes}
        save_json(args.output / 'refined_boxes.json', entity_data)
        save_json(args.output / 'box_refinements.json', {'coordinate_system': 'pixels_top_left', 'nodes': box_audit})
        print(f"框体轮廓校正：{sum(item['status'] == 'refined' for item in box_audit)} 个；"
              f"原始节点 JSON 保持不变。", flush=True)
    relevant = [n for n in nodes if n["bbox_global_px"][0] < x2 and n["bbox_global_px"][2] > x1 and n["bbox_global_px"][1] < y2 and n["bbox_global_px"][3] > y1]
    print(f"处理区域 {region}，节点 {len(relevant)}。", flush=True)
    threshold = image_threshold(image) if args.tile_size else None
    if args.tile_size:
        original_skeleton, tiles = prepare_tiled_skeleton(image, relevant, (x1, y1), args.min_line,
                                                          args.box_padding, args.tile_size, args.tile_overlap, threshold)
        seam_issues = find_unmatched_seam_endpoints(original_skeleton, (x1, y1), args.tile_size)
        save_json(args.output / 'tiles.json', {'image_size_px': list(size), 'region_global_px': region,
                   'tile_size_px': args.tile_size, 'tile_overlap_px': args.tile_overlap,
                   'threshold': threshold, 'tiles': tiles})
        print(f"分块完成：{len(tiles)} 个核心区域；拼接边界发现 {len(seam_issues)} 个待检查端点。", flush=True)
    else:
        original_skeleton = prepare_line_image(image, relevant, (x1, y1), args.min_line, args.box_padding)
        seam_issues = []
    Image.fromarray(original_skeleton).save(args.output / "line_skeleton_before_pruning.png")
    skeleton, removed = prune_short_spurs(original_skeleton, relevant, (x1, y1), args.spur_length, args.max_gap)
    skeleton, bridges, bridge_pixels = bridge_gaps(skeleton, relevant, (x1, y1), args.max_line_gap)
    Image.fromarray(skeleton).save(args.output / "line_skeleton.png")
    points = find_connection_points(skeleton, relevant, (x1, y1), args.max_gap)
    endpoint_map = {(p["point_global_px"][0] - x1, p["point_global_px"][1] - y1): p for p in points}
    traces = []
    for index, point in enumerate(points, 1):
        trace = trace_line(point, skeleton, endpoint_map, (x1, y1), bridge_pixels)
        if removed and trace["status"] == "reached_endpoint":
            original = trace_line(point, original_skeleton, endpoint_map, (x1, y1))
            trace["pruning_affected"] = (original["status"] != "reached_endpoint" or
                                        original.get("end_point_id") != trace["end_point_id"])
        traces.append(trace)
        if index % 50 == 0:
            print(f"追踪候选点 {index}/{len(points)}", flush=True)
    # Reverse traversal of the same pair of endpoints is one candidate path.
    kept, pairs = [], set()
    for trace in traces:
        if trace["status"] == "reached_endpoint":
            pair = tuple(sorted((trace["start_point_id"], trace["end_point_id"])))
            if pair in pairs:
                continue
            pairs.add(pair)
        kept.append(trace)
    result = {"coordinate_system": "pixels_top_left", "region_global_px": region,
              "note": "实验候选，连接点靠近节点不等于已确认连接；未识别箭头方向。",
              "removed_spurs": removed, "gap_hypotheses": bridges, "connection_points": points, "traces": kept}
    save_json(args.output / "candidates.json", result)
    if args.explore_branches or not args.candidates_only:
        hypotheses = branch_hypotheses(skeleton, points, region, bridges, bridge_pixels,
                                       args.max_junctions, args.max_branch_states)
        save_json(args.output / "branch_hypotheses.json", hypotheses)
        print(f"分叉路径假设 {len(hypotheses['traces'])} 条，均需复核。", flush=True)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for node in relevant:
        box = node["bbox_global_px"]
        draw.rectangle((box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1), outline="blue", width=3)
    for trace in kept:
        path = [(x - x1, y - y1) for x, y in trace["path_global_px"]]
        if len(path) > 1:
            draw.line(path, fill="lime" if trace["status"] == "reached_endpoint" and not trace['via_gap'] else "orange", width=5)
        if trace["status"] != "reached_endpoint":
            px, py = trace["position_global_px"]
            draw.ellipse((px-x1-8, py-y1-8, px-x1+8, py-y1+8), fill="orange")
    for p in points:
        px, py = p["point_global_px"]
        draw.ellipse((px-x1-6, py-y1-6, px-x1+6, py-y1+6), fill="red")
        draw.text((px-x1+8, py-y1), p["id"], fill="red", stroke_width=1, stroke_fill="white")
    marked.save(args.output / "candidates_marked.png")
    print(f"移除候选毛刺 {len(removed)}，候选点 {len(points)}，追踪记录 {len(kept)}，到达另一候选点 {sum(t['status'] == 'reached_endpoint' for t in kept)}。", flush=True)
    print("结果是候选，尚未判定箭头方向、节点真实连接及交叉处关系。", flush=True)
    if args.vision_review or args.junction_review:
        from model_client import ModelClient, ModelConfig
        config = json.loads(args.config.read_text(encoding="utf-8"))
        client = ModelClient(ModelConfig.model_validate(config["vision"]))
    if args.vision_review:
        review = review_paths(client, image, region, relevant, points, kept, args.output)
        save_json(args.output / "reviewed_edges.json", review)
        draw_reviewed_paths(image, region, review, args.output / "reviewed_marked.png")
        print(f"视觉复核完成：模型调用 {review['model_calls']} 次。", flush=True)
    if args.junction_review:
        review = review_junctions(client, image, region, kept, args.output)
        save_json(args.output / "junction_reviews.json", review)
        print(f"分叉复核完成：模型调用 {review['model_calls']} 次。", flush=True)
    if not args.candidates_only:
        from edge_graph import build_graph, infer_unknown_diamond_types
        from merge_edge_hypotheses import flag_hypotheses, merge

        # A second profile complements the first around node borders and text gaps.
        if args.tile_size:
            second, _ = prepare_tiled_skeleton(image, relevant, (x1, y1), args.min_line,
                                                24, args.tile_size, args.tile_overlap, threshold)
            second_seams = find_unmatched_seam_endpoints(second, (x1, y1), args.tile_size)
        else:
            second = prepare_line_image(image, relevant, (x1, y1), args.min_line, box_padding=24)
            second_seams = []
        second, _ = prune_short_spurs(second, relevant, (x1, y1), args.spur_length, protect_gap=45)
        second, second_bridges, second_pixels = bridge_gaps(second, relevant, (x1, y1), max_gap=260)
        second_points = find_connection_points(second, relevant, (x1, y1), max_gap=45)
        second_hypotheses = branch_hypotheses(second, second_points, region, second_bridges, second_pixels,
                                             args.max_junctions, args.max_branch_states)
        second_file = args.output / 'branch_hypotheses_profile_b.json'
        save_json(second_file, second_hypotheses)
        node_map = {node['id']: node for node in relevant}
        candidates = merge([args.output / 'branch_hypotheses.json', second_file], node_map)
        flag_hypotheses(candidates, image, node_map, attachment_threshold=32)
        graph, unresolved, evidence = build_graph(entity_data, candidates)
        graph = infer_unknown_diamond_types(graph, unresolved, box_audit)
        if not args.no_ocr:
            # Persist the topology before calling OCR so a provider failure
            # does not discard completed node and edge recognition.
            save_json(args.output / 'graph.json', {**graph, 'edge_text_status': 'pending'})
            from model_client import ModelClient, ModelConfig
            from text_extraction import extract_edge_text
            config = json.loads(args.config.read_text(encoding='utf-8'))
            ocr = ModelClient(ModelConfig.model_validate(config['ocr']))
            with Image.open(args.image) as original:
                graph, text_audit = extract_edge_text(ocr, original.convert('RGB'), graph,
                                                      args.output, resume=True, region=region)
            save_json(args.output / 'edge_text_review.json', text_audit)
            graph['edge_text_status'] = 'completed'
            print(f"边文字：关联 {len(text_audit['assigned'])} 处，待复核 {len(text_audit['unresolved'])} 处。", flush=True)
        else:
            graph['edge_text_status'] = 'not_requested'
        save_json(args.output / 'graph.json', graph)
        save_json(args.output / 'unresolved_edges.json', unresolved)
        save_json(args.output / 'edge_evidence.json', evidence)
        found_node_ids = {n['id'] for point in points + second_points for n in point['possible_nodes']}
        connected_node_ids = {edge[key] for edge in graph['edges'] for key in ('from', 'to')}
        save_json(args.output / 'unresolved_traces.json', {
            'coordinate_system': 'pixels_top_left', 'region_global_px': region,
            'profile_a': hypotheses['unresolved_traces'], 'profile_b': second_hypotheses['unresolved_traces'],
            'seam_endpoints_profile_a': seam_issues, 'seam_endpoints_profile_b': second_seams,
            'nodes_without_connection_candidates': [node['id'] for node in relevant if node['id'] not in found_node_ids],
            'nodes_without_output_edges': [node['id'] for node in relevant if node['id'] not in connected_node_ids],
            'entity_unresolved_nodes': unresolved_nodes,
            'note': '节点没有候选或输出边不代表原图没有关系；追踪中断和边界断点均需检查。'})
        print(f"节点关系 JSON 已生成：{len(graph['edges'])} 条候选有向边，{len(unresolved['edges'])} 条待复核。", flush=True)


if __name__ == "__main__":
    main()
