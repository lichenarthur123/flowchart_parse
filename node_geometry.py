"""Pixel coordinate utilities for visual deduplication."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw


def mark_target(image: Image.Image, point: list[float]) -> None:
    """Draw a visible marker whose center is exactly the candidate point."""
    x, y = point
    draw = ImageDraw.Draw(image)
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill='red', outline='white', width=2)


def point_box_distance(point: list[float], box: list[float]) -> float:
    """返回点到矩形的最短距离；点在框内时为 0。"""
    x, y = point
    x0, y0, x1, y1 = box
    return math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))


def local_to_global_box(box: object, crop_bbox: list[int], point: list[float],
                        point_tolerance: float = 0) -> list[float] | None:
    """校验局部框并换算为全局坐标；不修改模型返回的框。"""
    if not isinstance(box, list) or len(box) != 4:
        return None
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in box):
        return None
    x0, y0, x1, y1 = box
    width, height = crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    if point_tolerance < 0 or point_box_distance(point, box) > point_tolerance:
        return None
    # A box touching the crop edge could describe an incomplete node.
    if x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height:
        return None
    return [x0 + crop_bbox[0], y0 + crop_bbox[1], x1 + crop_bbox[0], y1 + crop_bbox[1]]


