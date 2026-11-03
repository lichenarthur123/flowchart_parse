"""Conservatively refine rectangular process and horizontal bar node boxes."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def refine_node_boxes(image: Image.Image, nodes: list[dict], origin: tuple[int, int],
                      margin: int = 150) -> tuple[list[dict], list[dict]]:
    """Use a nearby closed contour only if it contains the node point.

    Input boxes are never modified. Ambiguous candidates and partially
    cropped shapes are left unchanged.
    """
    gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    ox, oy = origin
    updated, audit = [], []
    for node in nodes:
        result = dict(node)
        box = node.get('bbox_global_px')
        center = node.get('center_global_px')
        item = {'id': node['id'], 'original_bbox_global_px': box,
                'bbox_global_px': box, 'status': 'unchanged'}
        node_type = node.get('type')
        if node_type not in ('process', 'fork', 'join', 'decision', 'merge', 'unknown') or not box or not isinstance(center, list) or len(center) != 2:
            item['reason'] = 'unsupported_type_or_missing_center'
            updated.append(result)
            audit.append(item)
            continue
        x1, y1, x2, y2 = box
        # Refuse partially cropped boxes: a contour cannot be confirmed there.
        if x1 < ox or y1 < oy or x2 > ox + width or y2 > oy + height:
            item['reason'] = 'partially_outside_region'
            updated.append(result)
            audit.append(item)
            continue
        l, t = max(0, int(x1-ox-margin)), max(0, int(y1-oy-margin))
        r, b = min(width, int(x2-ox+margin)), min(height, int(y2-oy+margin))
        if l >= r or t >= b:
            item['reason'] = 'empty_crop'
            updated.append(result)
            audit.append(item)
            continue
        candidates = []
        box_width, box_height = x2-x1, y2-y1
        if node_type in ('fork', 'join'):
            _, binary = cv2.threshold(gray[t:b, l:r], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(box_width*0.65)), 1)))
            _, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, 8)
            for x, y, w, h, area in stats[1:]:
                if not (0.65*box_width <= w <= 1.35*box_width and
                        max(3, 0.5*box_height) <= h <= 2*box_height and area/(w*h) >= 0.8):
                    continue
                candidate = [int(x+l+ox), int(y+t+oy), int(x+l+ox+w), int(y+t+oy+h)]
                if (abs((candidate[0]+candidate[2])/2-center[0]) > box_width*0.3 or
                    abs((candidate[1]+candidate[3])/2-center[1]) > max(30, box_height)):
                    continue
                difference = sum(abs(a-b) for a, b in zip(candidate, box))
                candidates.append((difference, -float(area/(w*h)), candidate))
        else:
            edges = cv2.Canny(gray[t:b, l:r], 35, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if node_type in ('decision', 'merge', 'unknown'):
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not (0.6*box_width <= w <= 1.4*box_width and 0.6*box_height <= h <= 1.4*box_height):
                    continue
                ratio = cv2.contourArea(contour) / (w*h)
                polygon = cv2.approxPolyDP(contour, 0.03*cv2.arcLength(contour, True), True)
                if len(polygon) != 4 or not 0.35 <= ratio <= 0.60 or not cv2.isContourConvex(polygon):
                    continue
                # Canny usually finds the inside edge of the diamond stroke.
                padding = max(2, round(min(w, h)*0.055))
                candidate = [x+l+ox-padding, y+t+oy-padding,
                             x+l+ox+w+padding, y+t+oy+h+padding]
                if not (ox <= candidate[0] < candidate[2] <= ox+width and oy <= candidate[1] < candidate[3] <= oy+height):
                    continue
                if not (candidate[0] < center[0] < candidate[2] and candidate[1] < center[1] < candidate[3]):
                    continue
                if any(other['id'] != node['id'] and isinstance(other.get('center_global_px'), list) and
                       len(other['center_global_px']) == 2 and candidate[0] < other['center_global_px'][0] < candidate[2] and
                       candidate[1] < other['center_global_px'][1] < candidate[3] for other in nodes):
                    continue
                if any(abs(a-b) > max(40, 0.3*max(box_width, box_height)) for a,b in zip(candidate, box)):
                    continue
                difference = sum(abs(a-b) for a,b in zip(candidate, box))
                candidates.append((difference, -ratio, candidate))
        if node_type == 'process':
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not (0.7*box_width <= w <= 1.3*box_width and 0.7*box_height <= h <= 1.3*box_height):
                    continue
                candidate = [x+l+ox, y+t+oy, x+l+ox+w, y+t+oy+h]
                if not (candidate[0]+0.05*w <= center[0] <= candidate[2]-0.05*w and
                        candidate[1]+0.05*h <= center[1] <= candidate[3]-0.05*h):
                    continue
                rectness = cv2.contourArea(contour) / (w*h)
                if rectness < 0.90:
                    continue
                # Never absorb a different node's center into the refined box.
                if any(other['id'] != node['id'] and isinstance(other.get('center_global_px'), list) and
                       len(other['center_global_px']) == 2 and
                       candidate[0] < other['center_global_px'][0] < candidate[2] and
                       candidate[1] < other['center_global_px'][1] < candidate[3] for other in nodes):
                    continue
                difference = sum(abs(a-b) for a, b in zip(candidate, box))
                candidates.append((difference, -rectness, candidate))
        if not candidates:
            item['reason'] = 'no_safe_shape_candidate'
        else:
            difference, negative_rectness, refined = min(candidates)
            result['bbox_global_px'] = refined
            item.update(bbox_global_px=refined, status='refined' if refined != box else 'unchanged',
                        max_offset_px=max(abs(a-b) for a, b in zip(refined, box)),
                        contour_rectness=round(-negative_rectness, 3))
            if node_type in ('decision', 'merge', 'unknown'):
                item['detected_shape'] = 'diamond'
        updated.append(result)
        audit.append(item)
    return updated, audit
