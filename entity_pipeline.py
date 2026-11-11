"""实体识别阶段：准备高清图并生成实体框体与文字。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pymupdf
from PIL import Image

from entity_extraction import process_image_entities, save_json
from model_client import ModelClient
from node_bounding_boxes import process_node_boxes


def prepare_image(source: Path, page: int, dpi: int, output: Path) -> dict:
    images = []

    def save_metadata(folder: Path, details: dict) -> None:
        with Image.open(folder / "image.png") as image:
            width, height = image.size
        details.update(id=folder.name, width=width, height=height,
                       coordinate_system="pixel_xyxy_top_left")
        save_json(folder / "metadata.json", details)
        images.append(details)

    if source.suffix.lower() == ".pdf":
        with pymupdf.open(source) as doc:
            if not 1 <= page <= len(doc):
                raise ValueError(f"页码须在 1 到 {len(doc)} 之间。")
            pdf_page = doc[page-1]
            regions = []
            for info in pdf_page.get_image_info(xrefs=True):
                rect = pymupdf.Rect(info["bbox"]) * pdf_page.rotation_matrix
                rect &= pdf_page.rect
                if not rect.is_empty and not rect.is_infinite:
                    regions.append((rect, info))
            if not regions:
                raise ValueError("该页没有嵌入图片，请先转换为高清图片。")
            regions.sort(key=lambda region: (region[0].y0, region[0].x0))
            for index, (rect, info) in enumerate(regions, 1):
                folder = output / f"image_{index:03d}"
                folder.mkdir()
                pix = pdf_page.get_pixmap(dpi=dpi, clip=rect, alpha=False)
                pix.save(folder / "image.png")
                details = {"source": str(source), "page": page, "dpi": dpi,
                           "embedded_image_size": [info["width"], info["height"]]}
                del pix
                save_metadata(folder, details)
    else:
        folder = output / "image_001"
        folder.mkdir()
        with Image.open(source) as original:
            original.convert("RGB").save(folder / "image.png")
        save_metadata(folder, {"source": str(source), "page": None})
    metadata = {"source": str(source), "page": page if source.suffix.lower() == ".pdf" else None,
                "images": images}
    save_json(output / "metadata.json", metadata)
    return metadata


def choose_grid(metadata: dict, tile_max_size: int, overlap_x: float, overlap_y: float,
                cols: int | None, rows: int | None) -> tuple[int, int]:
    if (cols is None) != (rows is None):
        raise ValueError("--cols 和 --rows 必须一起提供。")

    def count(length: int, overlap: float, target: int) -> int:
        if length <= tile_max_size:
            return 1
        value = 1
        while math.ceil(length / (value-(value-1)*overlap/100)) > min(tile_max_size, target):
            value += 1
        return value

    width = max(item["width"] for item in metadata["images"])
    height = max(item["height"] for item in metadata["images"])
    cols = cols or count(width, overlap_x, 1800)
    rows = rows or count(height, overlap_y, 1700)
    for item in metadata["images"]:
        if not (1 <= cols <= item["width"] and 1 <= rows <= item["height"]):
            raise ValueError("切片数须为正整数，且不能超过图片像素数。")
        tile_width = item["width"] if cols == 1 else math.ceil(item["width"]/(cols-(cols-1)*overlap_x/100))
        tile_height = item["height"] if rows == 1 else math.ceil(item["height"]/(rows-(rows-1)*overlap_y/100))
        if max(tile_width, tile_height) > tile_max_size:
            raise ValueError("切片超过尺寸限制，请增加 --cols 或 --rows。")
    return cols, rows


def recognize_entities(source: Path, work_dir: Path, vision: ModelClient, ocr: ModelClient | None,
                       page: int, dpi: int, cols: int | None, rows: int | None,
                       overlap_x: float = 20, overlap_y: float = 20, tile_max_size: int = 2000,
                       dedup_window: int = 2000, dedup_overlap: float = 25,
                       resume: bool = False, workers: int = 4) -> tuple[Path, dict]:
    if dpi < 1 or tile_max_size < 1 or dedup_window < 1:
        raise ValueError("DPI、切片尺寸和去重窗口必须为正数。")
    if not 0 <= overlap_x < 100 or not 0 <= overlap_y < 100 or not 0 <= dedup_overlap < 100:
        raise ValueError("重叠比例必须在 [0,100) 内。")
    metadata_path = work_dir / "metadata.json"
    if resume:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        metadata = prepare_image(source, page, dpi, work_dir)
    if len(metadata["images"]) != 1:
        raise ValueError("当前流程每次只支持一张高清图；PDF 页面包含多张图片时请分别处理。")
    cols, rows = choose_grid(metadata, tile_max_size, overlap_x, overlap_y, cols, rows)
    image_metadata = metadata["images"][0]
    folder = work_dir / image_metadata["id"]
    args = SimpleNamespace(cols=cols, rows=rows, overlap_x=overlap_x, overlap_y=overlap_y,
                           tile_max_size=tile_max_size, dedup_window=dedup_window,
                           dedup_overlap=dedup_overlap, resume=resume, split_only=False,
                           workers=workers)
    process_image_entities(folder / "image.png", folder, image_metadata, args, vision)
    process_node_boxes(vision, folder / "image.png", folder, folder / "nodes.json",
                       resume=resume, ocr_client=ocr, workers=workers)
    boxes = json.loads((folder / "boxes.json").read_text(encoding="utf-8"))
    return folder, boxes
