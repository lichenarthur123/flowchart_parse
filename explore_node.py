"""以局部高清截图探索实体的一度出向关系。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from model_client import ModelClient


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def node_labels(nodes: list[dict]) -> dict[str, str]:
    labels = {}
    used = set()
    for index, node in enumerate(nodes, 1):
        match = re.search(r"N\d+", node.get("text") or "", re.IGNORECASE)
        label = match.group(0).upper() if match else f"E{index:03d}"
        if label in used:
            label = node["id"]
        used.add(label)
        labels[node["id"]] = label
    return labels


def replace_node_labels(text: str, labels: dict[str, str]) -> str:
    by_label = {label: node_id for node_id, label in labels.items()}
    text = re.sub(r"/[NE]\d+", "", text, flags=re.IGNORECASE)
    return re.sub(r"(?<![A-Za-z0-9])[NE]\d+(?![A-Za-z0-9])",
                  lambda match: by_label.get(match.group(0).upper(), ""), text, flags=re.IGNORECASE)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def mark_nodes(image: Image.Image, nodes: list[dict], labels: dict[str, str], entrypoint: str) -> None:
    draw = ImageDraw.Draw(image)
    label_font = font(24)
    for node in nodes:
        x, y = node["center_global_px"]
        current = node["id"] == entrypoint
        color = "#e11d48" if current else "#2563eb"
        radius = 14 if current else 9
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="white", width=2)
        draw.text((x+radius+5, y-18), labels[node["id"]], fill=color, font=label_font,
                  stroke_width=2, stroke_fill="white")


def centered_window(point: list[float], width: int, height: int, image_size: tuple[int, int]) -> list[int]:
    image_width, image_height = image_size
    width, height = min(width, image_width), min(height, image_height)
    left = max(0, min(image_width-width, round(point[0]-width/2)))
    top = max(0, min(image_height-height, round(point[1]-height/2)))
    return [left, top, left+width, top+height]


def move_window(window: list[int], action: dict, image_size: tuple[int, int]) -> list[int]:
    image_width, image_height = image_size
    direction = str(action.get("direction", "")).lower()
    kind = str(action.get("type", "")).lower()
    x1, y1, x2, y2 = window
    if kind == "expand":
        amount = 400
        dx = amount if direction in ("left", "right", "all") else 0
        dy = amount if direction in ("up", "down", "all") else 0
        left = x1-dx if direction in ("left", "all") else x1
        right = x2+dx if direction in ("right", "all") else x2
        top = y1-dy if direction in ("up", "all") else y1
        bottom = y2+dy if direction in ("down", "all") else y2
        if direction not in ("left", "right", "up", "down", "all"):
            left, top, right, bottom = x1-amount, y1-amount, x2+amount, y2+amount
    else:
        dx = round((x2-x1)*0.7) if direction == "right" else -round((x2-x1)*0.7) if direction == "left" else 0
        dy = round((y2-y1)*0.7) if direction == "down" else -round((y2-y1)*0.7) if direction == "up" else 0
        left, top, right, bottom = x1+dx, y1+dy, x2+dx, y2+dy
    left = max(0, min(image_width-1, left))
    top = max(0, min(image_height-1, top))
    right = min(image_width, max(left+1, right))
    bottom = min(image_height, max(top+1, bottom))
    return [left, top, right, bottom]


def local_path(path: object, window: list[int]) -> list[list[float]] | None:
    if not isinstance(path, list) or len(path) < 2:
        return None
    result = []
    width, height = window[2]-window[0], window[3]-window[1]
    for point in path:
        if (not isinstance(point, list) or len(point) != 2 or
                any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in point)):
            return None
        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            return None
        result.append([x+window[0], y+window[1]])
    return result


def draw_response_paths(image: Image.Image, response: dict, window: list[int]) -> None:
    draw = ImageDraw.Draw(image)
    confirmed_clues = {r.get("clue_id") for r in response.get("relations", [])
                       if isinstance(r, dict) and r.get("direction") == "entrypoint_to_other"}
    for clue in response.get("clues", []):
        if not isinstance(clue, dict) or not isinstance(clue.get("clue_id"), str):
            continue
        path = local_path(clue.get("path_local_px"), window)
        if path is None:
            continue
        color = "#16a34a" if clue["clue_id"] in confirmed_clues else "#f59e0b"
        draw.line([tuple(p) for p in path], fill=color, width=6)
        x, y = path[len(path)//2]
        draw.text((x+8, y+8), clue["clue_id"], fill=color, font=font(24), stroke_width=2, stroke_fill="white")


def build_prompt(entry: dict, nodes: list[dict], labels: dict[str, str],
                 confirmed: list[dict], history: list[dict], window: list[int]) -> str:
    nearby = [n for n in nodes if n["id"] == entry["id"] or
              (window[0]-200 <= n["center_global_px"][0] <= window[2]+200 and
               window[1]-200 <= n["center_global_px"][1] <= window[3]+200)]
    catalog = [{"id": n["id"], "label": labels[n["id"]], "text": n.get("text", ""),
                "center_global_px": n["center_global_px"]} for n in nearby]
    previous = [{"window_global_px": step["window"], "observation": step["response"].get("observation"),
                 "next_action": step["response"].get("next_action")} for step in history[-3:]]
    return f'''你在阅读一张流程图，只探索当前 entrypoint 的一度直接关系。
entrypoint 的实体 ID 是 {entry["id"]}，图上标号是 {labels[entry["id"]]}，文字为 {entry.get("text", "")}，全局中心点为 {entry["center_global_px"]}。
蓝色标点是普通节点，红色标点是当前 entrypoint；绿色线是已确认线索，橙色线是仍在探索的线索。
所有节点的真实 ID、图上标号、文字和全局坐标：{json.dumps(catalog, ensure_ascii=False)}
已确认关系：{json.dumps(confirmed, ensure_ascii=False)}
此前阅读历史：{json.dumps(previous, ensure_ascii=False)}

只探索经过当前 entrypoint 的一度直接关系。优先根据可见连线和箭头判断；没有明确箭头时，可以结合连线和图意理解关系，但必须在 evidence 中说明。
若一度关系完整可见，记录关系；若对端或关系不可见，为一条仍需追踪的线索分配 clue_id，并选择滑动或扩大方向。
每确认一条新关系后应返回 entrypoint。已确认关系不要重复输出。只有 entrypoint 出向关系会进入最终产出，但入边也应观察。
可以在 clues 中提交明确看到的线索路径，path_local_px 使用当前截图像素坐标，不要提交不确定的路径。
只返回 JSON 对象：
{{"observation":"当前看到的内容",
 "clues":[{{"clue_id":"clue_001","path_local_px":[[100,100],[200,100]],"status":"exploring"}}],
 "relations":[{{"other_node_id":"目标真实ID","direction":"entrypoint_to_other","clue_id":"clue_001","relation":"entrypoint 与目标的关系","evidence":"视觉或图意依据"}}],
 "next_action":{{"type":"move|expand|return|finish","direction":"left|right|up|down|all|none","clue_id":"clue_001","reason":"为什么这样移动"}}}}
relation 和 evidence 中提及任何节点时必须使用真实 ID，禁止使用 N 或 E 开头的图上标号。
确认 entrypoint 没有其他未阅读的一度关系后，next_action.type 返回 finish。'''


def explore_relations(client: ModelClient, image_path: Path, entities_path: Path, work_dir: Path,
                      max_steps: int = 8, resume: bool = False) -> list[dict]:
    if max_steps < 1:
        raise ValueError("每个实体的探索步数必须大于 0。")
    work_dir.mkdir(parents=True, exist_ok=True)
    nodes = json.loads(entities_path.read_text(encoding="utf-8"))["nodes"]
    labels = node_labels(nodes)
    progress_path = work_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if resume and progress_path.is_file() else {}
    relations = progress.get("relations", [])
    completed = set(progress.get("completed", []))
    relation_keys = {(r["from"], r["to"]) for r in relations}
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    for entry in nodes:
        if entry["id"] in completed:
            print(f"复用已完成的 entrypoint：{entry['id']}", flush=True)
            continue
        print(f"开始处理 entrypoint：{entry['id']}", flush=True)
        marked = original.copy()
        mark_nodes(marked, nodes, labels, entry["id"])
        initial = centered_window(entry["center_global_px"], 1100, 900, marked.size)
        window = initial
        confirmed = []
        confirmed_targets = set()
        history = []
        for _ in range(max_steps):
            screenshot = work_dir / "current.png"
            marked.crop(tuple(window)).save(screenshot)
            response = client.ask_json(build_prompt(entry, nodes, labels, confirmed, history, window),
                                       [screenshot], stage="single_node_relation_exploration")
            history.append({"window": window, "response": response})
            draw_response_paths(marked, response, window)
            new_relation = False
            for relation in response.get("relations", []):
                if not isinstance(relation, dict):
                    continue
                other_id = relation.get("other_node_id")
                if other_id not in labels or other_id == entry["id"] or relation.get("direction") != "entrypoint_to_other":
                    continue
                if other_id in confirmed_targets:
                    continue
                confirmed_targets.add(other_id)
                relation_text = str(relation.get("relation") or f"{entry['id']} 指向 {other_id}")
                confirmed.append({
                    "id": "",
                    "from": entry["id"],
                    "to": other_id,
                    "relation": replace_node_labels(relation_text, labels),
                    "evidence": replace_node_labels(str(relation.get("evidence") or ""), labels),
                })
                new_relation = True
            action = response.get("next_action") if isinstance(response.get("next_action"), dict) else {}
            action_type = str(action.get("type", "finish")).lower()
            if action_type == "finish":
                break
            if new_relation or action_type == "return":
                window = initial
            elif action_type in ("move", "expand"):
                window = move_window(window, action, marked.size)
            else:
                break
        added = 0
        for relation in confirmed:
            key = (relation["from"], relation["to"])
            if key not in relation_keys:
                relation_keys.add(key)
                relations.append(relation)
                added += 1
        completed.add(entry["id"])
        save_json(progress_path, {"relations": relations, "completed": sorted(completed)})
        print(f"完成 entrypoint：{entry['id']}，新增 {added} 条关系", flush=True)
    for index, relation in enumerate(relations, 1):
        relation["id"] = f"rel_{index:03d}"
    save_json(progress_path, {"relations": relations, "completed": sorted(completed)})
    return relations
