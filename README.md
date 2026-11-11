# 高清流程图实体与关系识别

统一完成两个阶段：

1. 实体识别：节点定位、去重、框体坐标和框内 OCR。
2. 关系抽取：逐个实体截取局部高清图，通过滑动或扩大窗口探索一度出向关系。

## 安装与配置

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp config.example.json config.json
```

在 `config.json` 中配置 OpenAI 兼容的 `vision` 和 `ocr` 模型。`vision` 用于实体定位、框体和关系理解，`ocr` 用于框内文字。配置文件包含密钥，请勿提交。

```json
{
  "vision": {
    "base_url": "https://YOUR_PROVIDER/v1",
    "model": "YOUR_VISION_MODEL",
    "api_key": "YOUR_API_KEY",
    "timeout_seconds": 300,
    "max_tokens": 32768
  },
  "ocr": {
    "base_url": "https://YOUR_PROVIDER/v1",
    "model": "YOUR_OCR_MODEL",
    "api_key": "YOUR_API_KEY",
    "timeout_seconds": 300,
    "max_tokens": 32768
  }
}
```

所有模型请求统一使用对应配置中的 `max_tokens`，不再按处理阶段设置较小上限；可根据模型支持范围调整。也可以使用 `api_key_env` 代替明文 `api_key`。

## 运行

```bash
source .venv/bin/activate
python3 run_workflow.py 'output/复杂流程图测试/复杂采购流程图_高清.png' \
  --output runs/full-workflow
```

支持参数：

- `--page N`：PDF 页码，从 1 开始。
- `--dpi N`：PDF 截图 DPI，默认 300。
- `--cols N --rows N`：手动设置实体识别网格；省略时自动计算。
- `--no-ocr`：跳过框内文字识别。
- `--max-steps N`：每个实体的最大关系探索步数，默认 8。
- `--workers N`：实体切片、去重窗口、框体和 OCR 的并发数，默认 4。
- `--resume`：从中断位置继续，复用已有缓存。

实体识别中的切片定位、去重窗口、框体和 OCR 默认使用 4 路并发；同一个 entrypoint 的关系探索保持串行，因为下一步窗口依赖上一步结果。关系探索 Prompt 只携带当前窗口附近节点和最近三步历史。

中断后使用相同命令并添加 `--resume`：

```bash
python3 run_workflow.py 'output/复杂流程图测试/复杂采购流程图_高清.png' \
  --output runs/full-workflow --resume
```

任务成功后中间缓存会清理，已完成任务无需续跑。当前每次处理一张高清图；PDF 页面包含多张图片时请分别处理，纯矢量页面请先转换成图片。

## 输出

输出目录固定为两个文件夹：

```text
runs/full-workflow/
├── final/
│   ├── entities.json
│   └── relations.json
└── intermediate/
    ├── global_marked.png
    └── global_boxes_marked.png
```

- `final/entities.json`：实体 ID、类型、中心点、框体坐标和框内文字。
- `final/relations.json`：实体出向关系。
- `intermediate/global_marked.png`：实体中心点标记图。
- `intermediate/global_boxes_marked.png`：实体框体标记图。

切片、OCR 缓存、关系探索截图和进度文件在运行时保存在 `intermediate/` 的子目录中，成功后自动删除；运行失败时保留，便于排查和续跑。不生成 HTML 或模型用量文件。

每条关系只包含五个字段：

```json
{
  "id": "rel_001",
  "from": "起始节点实体ID",
  "to": "结束节点实体ID",
  "relation": "关系表述",
  "evidence": "判断依据"
}
```

`relation` 和 `evidence` 中的节点统一使用真实实体 ID。输出仍需人工复核，不能视为已验证的完整实体关系清单。

## 代码结构

- `run_workflow.py`：唯一运行入口，负责目录管理和两阶段串联。
- `entity_pipeline.py`：高清图准备、自动网格和实体识别。
- `entity_extraction.py`：节点定位、坐标换算和去重。
- `node_bounding_boxes.py`：实体框体和 OCR 调度。
- `node_geometry.py`：坐标与框体校验。
- `text_extraction.py`：框内 OCR 及缓存。
- `explore_node.py`：局部截图、窗口滑动和一度关系探索。
- `model_client.py`：OpenAI 兼容模型调用。
- `usage_tracking.py`：模型请求记录接口。
