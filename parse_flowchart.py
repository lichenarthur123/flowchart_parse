#!/usr/bin/env python3
"""准备 PDF 图片并运行流程图节点识别。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pymupdf
from PIL import Image

from model_client import ModelClient, ModelConfig
from entity_extraction import process_image_entities, process_single_region
from node_bounding_boxes import process_node_boxes
from usage_tracking import ENV_VAR, summarize


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_edge_stage(image: Path, folder: Path, unresolved_candidates: Path | None, resume: bool,
                   config: Path, no_ocr: bool) -> Path:
    """Run the full-image edge stage without overwriting interrupted results."""
    boxes = folder / 'boxes.json'
    script = Path(__file__).with_name('edge_extraction.py')
    files = [image, boxes, script, Path(__file__).with_name('edge_graph.py'),
             Path(__file__).with_name('merge_edge_hypotheses.py'),
             Path(__file__).with_name('node_box_refinement.py'),
             Path(__file__).with_name('text_extraction.py'), Path(__file__).with_name('model_client.py')]
    if not no_ocr:
        files.append(config)
    unresolved_boxes = folder / 'unresolved_boxes.json'
    for path in (unresolved_boxes, unresolved_candidates):
        if path is not None and path.is_file():
            files.append(path)
    inputs = {str(path.resolve()): file_sha256(path) for path in files}
    manifest_path = folder / 'edge_run_manifest.json'
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        previous_name = manifest.get('output')
        if not isinstance(previous_name, str) or Path(previous_name).name != previous_name or not previous_name.startswith('edges'):
            raise ValueError('边识别运行记录无效。')
        previous = folder / previous_name
        required = ['graph.json', 'edge_evidence.json', 'unresolved_edges.json', 'unresolved_traces.json']
        if not no_ocr:
            required.append('edge_text_review.json')
        if manifest.get('inputs') == inputs and manifest.get('no_ocr') == no_ocr and all((previous / name).is_file() for name in required):
            print(f'复用已完成的边识别：{previous}', flush=True)
            return previous
    target = folder / 'edges'
    counter = 1
    while target.exists():
        target = folder / f'edges_retry_{counter:03d}'
        counter += 1
    save_json(manifest_path, {'inputs': inputs, 'output': target.name, 'no_ocr': no_ocr})
    command = [sys.executable, str(script), str(image), str(boxes), '--output', str(target), '--config', str(config)]
    if no_ocr:
        command.append('--no-ocr')
    if unresolved_candidates is not None and unresolved_candidates.is_file():
        command.extend(['--unresolved-nodes', str(unresolved_candidates)])
    print(f'开始全图边识别：{target}', flush=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'边识别失败，退出码 {exc.returncode}；已有结果保留在 {target}。') from None
    return target


def prepare_image(source: Path, page: int, dpi: int, out: Path) -> dict:
    """截取指定页中显示的图片区域，坐标以各截图左上角为原点。"""
    images = []

    def save_image_metadata(folder: Path, details: dict) -> None:
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
            pdf_page = doc[page - 1]
            # Image bounding boxes use the unrotated page; get_pixmap clips the displayed page.
            regions = []
            for info in pdf_page.get_image_info(xrefs=True):
                rect = pymupdf.Rect(info["bbox"]) * pdf_page.rotation_matrix
                rect &= pdf_page.rect
                if not rect.is_empty and not rect.is_infinite:
                    regions.append((rect, info))
            if not regions:
                raise ValueError("该页没有检测到嵌入图片；矢量流程图或页面截图请先转换为图片。")
            regions.sort(key=lambda region: (region[0].y0, region[0].x0))
            for index, (rect, info) in enumerate(regions, start=1):
                folder = out / f"image_{index:03d}"
                folder.mkdir()
                pix = pdf_page.get_pixmap(dpi=dpi, clip=rect, alpha=False)
                pix.save(folder / "image.png")
                # Rounding to device pixels can shift the rendered crop by a fraction of a point.
                pixel_to_page = [72 / dpi, 0, 0, 72 / dpi, pix.x * 72 / dpi, pix.y * 72 / dpi]
                details = {"source": str(source), "page": page, "dpi": dpi,
                           "page_bbox_pt": list(rect), "page_rotation": pdf_page.rotation,
                           "pixel_to_page_matrix": pixel_to_page,
                           "embedded_image_size": [info["width"], info["height"]]}
                del pix
                save_image_metadata(folder, details)
    else:
        folder = out / "image_001"
        folder.mkdir()
        with Image.open(source) as original:
            original.convert("RGB").save(folder / "image.png")
        save_image_metadata(folder, {"source": str(source), "page": None})
    metadata = {"source": str(source), "page": page if source.suffix.lower() == ".pdf" else None,
                "status": "prepared", "images": images}
    save_json(out / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="流程图节点识别：图片准备、分块识别和去重")
    parser.add_argument("input", type=Path)
    parser.add_argument("--page", type=int, default=1, help="PDF 页码，从 1 开始")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 图片区域截图分辨率，默认 300")
    parser.add_argument("--output", type=Path, default=Path("runs/sample"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--prepare-only", action="store_true", help="仅准备图片，不调用模型")
    parser.add_argument("--split-only", action="store_true", help="仅按网格切图，不调用模型")
    parser.add_argument("--locate-nodes", action="store_true", help="只识别指定局部图中的节点和中心点")
    parser.add_argument("--boxes-from-nodes", type=Path, help="根据节点坐标 JSON 识别节点框")
    parser.add_argument("--bbox-crop-size", type=int, default=1200, help="节点框识别的高清截图最大边长，默认 1200")
    parser.add_argument("--retry-unresolved-boxes", action="store_true", help="继续节点框任务时重新识别未确定的框")
    parser.add_argument("--with-edges", action="store_true", help="全图节点完成后继续识别节点框和全图边")
    parser.add_argument("--no-ocr", action="store_true", help="跳过节点和边文字识别")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="高清图中的裁剪区域，左上到右下的像素坐标")
    parser.add_argument("--cols", type=int, help="横向等分数")
    parser.add_argument("--rows", type=int, help="纵向等分数")
    parser.add_argument("--overlap-x", type=float, default=20, help="横向重叠百分比，默认 20")
    parser.add_argument("--overlap-y", type=float, default=20, help="纵向重叠百分比，默认 20")
    parser.add_argument("--tile-max-size", type=int, default=2000, help="模型输入局部图最大边长，默认 2000 像素")
    parser.add_argument("--dedup-window", type=int, default=2000, help="去重滑窗边长（高清图像素），默认 2000")
    parser.add_argument("--dedup-overlap", type=float, default=25, help="去重滑窗重叠百分比，默认 25")
    parser.add_argument("--resume", action="store_true", help="继续未完成的网格节点识别，复用已完成分块")
    args = parser.parse_args()
    usage_path = None
    try:
        if args.dpi <= 0:
            raise ValueError("dpi 必须大于 0。")
        if sum([args.prepare_only, args.split_only, args.locate_nodes, args.boxes_from_nodes is not None]) != 1:
            raise ValueError("请选择一个运行模式：--prepare-only、--split-only、--locate-nodes 或 --boxes-from-nodes。")
        if args.bbox_crop_size <= 0:
            raise ValueError("--bbox-crop-size 必须大于 0。")
        if args.retry_unresolved_boxes and (not args.resume or args.boxes_from_nodes is None):
            raise ValueError("--retry-unresolved-boxes 需与 --boxes-from-nodes 和 --resume 一起使用。")
        if args.dedup_window <= 0 or not 0 <= args.dedup_overlap < 100:
            raise ValueError("去重滑窗边长须大于 0，重叠百分比须在 [0, 100) 内。")
        if args.tile_max_size <= 0:
            raise ValueError("--tile-max-size 必须大于 0。")
        if (args.cols is None) != (args.rows is None):
            raise ValueError("--cols 和 --rows 必须一起提供。")
        grid_requested = args.cols is not None
        if args.crop is not None and grid_requested:
            raise ValueError("--crop 和网格等分参数不能同时使用。")
        if args.split_only and not grid_requested:
            raise ValueError("--split-only 需要 --cols 和 --rows。")
        if args.locate_nodes and args.crop is None and not grid_requested:
            raise ValueError("--locate-nodes 需要 --crop，或同时提供 --cols 和 --rows。")
        if args.with_edges and not ((args.locate_nodes and grid_requested) or args.boxes_from_nodes is not None):
            raise ValueError('--with-edges 需与全图网格节点识别或 --boxes-from-nodes 一起使用。')
        if args.crop is not None and not args.locate_nodes:
            raise ValueError("--crop 需与 --locate-nodes 一起使用。")
        if grid_requested and not (args.split_only or args.locate_nodes):
            raise ValueError("--cols 和 --rows 需与 --split-only 或 --locate-nodes 一起使用。")
        if args.resume and not ((args.locate_nodes and grid_requested) or args.boxes_from_nodes is not None):
            raise ValueError("--resume 只支持网格节点识别或节点框识别。")
        source = args.input.resolve(strict=True)
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise ValueError("输出目录非空，请使用新的 --output 路径，避免覆盖已有结果。")
        clients = {}
        if not (args.prepare_only or args.split_only):
            config = json.loads(args.config.read_text(encoding="utf-8"))
            clients["vision"] = ModelClient(ModelConfig.model_validate(config["vision"]))
            if not args.no_ocr and (args.with_edges or args.boxes_from_nodes is not None):
                clients["ocr"] = ModelClient(ModelConfig.model_validate(config["ocr"]))
        manifest = None
        if args.locate_nodes and grid_requested:
            vision = clients["vision"].config
            manifest = {"source": str(source), "source_sha256": file_sha256(source),
                        "page": args.page if source.suffix.lower() == ".pdf" else None,
                        "dpi": args.dpi if source.suffix.lower() == ".pdf" else None,
                        "cols": args.cols, "rows": args.rows, "overlap_x": args.overlap_x,
                        "overlap_y": args.overlap_y, "tile_max_size": args.tile_max_size,
                        "dedup_window": args.dedup_window, "dedup_overlap": args.dedup_overlap,
                        "workflow_version": 13,
                        "vision": {"base_url": vision.base_url, "model": vision.model,
                                   "max_tokens": vision.max_tokens, "json_mode": vision.json_mode,
                                   "enable_thinking": vision.enable_thinking}}
        if args.boxes_from_nodes is not None:
            node_source = args.boxes_from_nodes.resolve(strict=True)
            vision = clients["vision"].config
            manifest = {"stage": "node_boxes_v1", "validation_version": 2,
                        "source": str(source), "source_sha256": file_sha256(source),
                        "nodes": str(node_source), "nodes_sha256": file_sha256(node_source),
                        "page": args.page if source.suffix.lower() == ".pdf" else None,
                        "dpi": args.dpi if source.suffix.lower() == ".pdf" else None,
                        "bbox_crop_size": args.bbox_crop_size,
                        "vision": {"base_url": vision.base_url, "model": vision.model,
                                   "max_tokens": vision.max_tokens, "json_mode": vision.json_mode,
                                   "enable_thinking": vision.enable_thinking}}
        if args.resume:
            manifest_path = args.output / "run_manifest.json"
            if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("无法继续：运行记录不存在，或输入文件、网格参数、模型配置已改变。")
            metadata = json.loads((args.output / "metadata.json").read_text(encoding="utf-8"))
            print(f"继续已有任务：{args.output.resolve()}", flush=True)
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            metadata = prepare_image(source, args.page, args.dpi, args.output)
            if manifest is not None:
                save_json(args.output / "run_manifest.json", manifest)
            print(f"图片准备完成：共 {len(metadata['images'])} 张，输出目录：{args.output.resolve()}", flush=True)
        usage_path = (args.output / 'model_usage.jsonl').resolve()
        os.environ[ENV_VAR] = str(usage_path)
        if args.prepare_only:
            return
        if args.boxes_from_nodes is not None:
            if len(metadata['images']) != 1:
                raise ValueError("节点坐标须对应一张图片；当前页提取出了多张图片，请先提供目标图片。")
            folder = args.output / metadata['images'][0]['id']
            process_node_boxes(clients['vision'], folder / 'image.png', folder, args.boxes_from_nodes,
                               args.bbox_crop_size, args.resume, args.retry_unresolved_boxes, clients.get('ocr'))
            if args.with_edges:
                run_edge_stage(folder / 'image.png', folder,
                               args.boxes_from_nodes.parent / 'unresolved_candidates.json', args.resume,
                               args.config, args.no_ocr)
            return
        for image_metadata in metadata["images"]:
            folder = args.output / image_metadata["id"]
            image = folder / "image.png"
            if grid_requested:
                process_image_entities(image, folder, image_metadata, args, clients.get("vision"))
                if args.with_edges:
                    process_node_boxes(clients['vision'], image, folder, folder / 'nodes.json',
                                       args.bbox_crop_size, args.resume, ocr_client=clients.get('ocr'))
                    run_edge_stage(image, folder, folder / 'unresolved_candidates.json', args.resume,
                                   args.config, args.no_ocr)
                continue
            if args.locate_nodes:
                process_single_region(image, folder, args, clients["vision"])
                continue
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        if usage_path is not None:
            try:
                save_json(args.output / 'model_usage_summary.json', summarize(usage_path))
            except (OSError, ValueError) as exc:
                print(f'模型用量汇总失败：{exc}', file=sys.stderr)


if __name__ == "__main__":
    main()
