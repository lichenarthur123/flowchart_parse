#!/usr/bin/env python3
"""统一执行实体识别和实体关系抽取。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from entity_pipeline import recognize_entities
from explore_node import explore_relations
from model_client import ModelClient, ModelConfig
from usage_tracking import ENV_VAR


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4, help="实体识别并发数，默认 4")
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    final = output / "final"
    intermediate = output / "intermediate"
    entities_path = final / "entities.json"
    relations_path = final / "relations.json"
    if entities_path.is_file() and relations_path.is_file():
        print(f"任务已完成：{output}")
        return
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("输出目录非空，请使用新目录或 --resume。")
    final.mkdir(parents=True, exist_ok=True)
    intermediate.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    vision = ModelClient(ModelConfig.model_validate(config["vision"]))
    ocr = None if args.no_ocr else ModelClient(ModelConfig.model_validate(config["ocr"]))
    os.environ.pop(ENV_VAR, None)
    try:
        entity_dir = intermediate / "entities" / "image_001"
        if args.resume and entities_path.is_file() and (entity_dir / "image.png").is_file():
            boxes = json.loads(entities_path.read_text(encoding="utf-8"))
        else:
            entity_dir, boxes = recognize_entities(
                source, intermediate / "entities", vision, ocr, args.page, args.dpi,
                args.cols, args.rows, resume=args.resume, workers=args.workers)
            save_json(entities_path, boxes)
        shutil.copy2(entity_dir / "global_marked.png", intermediate / "global_marked.png")
        shutil.copy2(entity_dir / "global_boxes_marked.png", intermediate / "global_boxes_marked.png")
        relations = explore_relations(vision, entity_dir / "image.png", entities_path,
                                      intermediate / "relations", args.max_steps, args.resume)
        save_json(relations_path, relations)
        shutil.rmtree(intermediate / "entities", ignore_errors=True)
        shutil.rmtree(intermediate / "relations", ignore_errors=True)
        print(f"实体：{len(boxes['nodes'])}，关系：{len(relations)}")
        print(f"最终输出：{final}")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        print(f"中间状态已保留：{intermediate}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
