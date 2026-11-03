#!/usr/bin/env bash
# 从图片或 PDF 指定页运行完整流程；默认使用 PATH 中的 python3。
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  ./test_flowchart.sh 图片或PDF路径 [选项]

选项：
  --page N            PDF 页码，从 1 开始；默认 1
  --output 目录       结果目录；默认在项目 runs 下创建新目录
  --config 文件       模型配置；默认项目中的 config.json
  --dpi N             PDF 截图 DPI；默认 300
  --cols N            横向切片数；默认按图片尺寸计算
  --rows N            纵向切片数；默认按图片尺寸计算
  --overlap-x 百分比  横向切片重叠比例；默认 20
  --overlap-y 百分比  纵向切片重叠比例；默认 20
  --tile-max-size N   单张节点切片最大边长；默认 2000 像素
  --no-ocr            跳过节点和边文字识别
  --resume            继续已有任务，必须同时提供原 --output
  --dry-run           检查输入和网格，打印命令但不调用模型
  -h, --help          显示帮助

示例：
  ./test_flowchart.sh 'output/复杂流程图测试/复杂采购流程图_高清.png'
  ./test_flowchart.sh 'output/pdf/采购流程解析测试_图片版.pdf' --page 4
  ./test_flowchart.sh '图.png' --cols 3 --rows 6 --output runs/my-test
  ./test_flowchart.sh '图.png' --cols 3 --rows 6 --output runs/my-test --resume

PDF 指定页须包含嵌入图片；矢量页面请先转换成图片。
请先激活包含 requirements.txt 依赖的 Python 环境，并配置 config.json。
EOF
}

if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
if [[ "$1" == '-h' || "$1" == '--help' ]]; then usage; exit 0; fi
caller_dir=$PWD
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
source_file=$1
shift
case "$source_file" in /*) ;; *) source_file="$caller_dir/$source_file" ;; esac
page=1
dpi=300
cols=''
rows=''
overlap_x=20
overlap_y=20
tile_max=2000
config="$project_dir/config.json"
output=''
no_ocr=0
resume=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --page|--output|--config|--dpi|--cols|--rows|--overlap-x|--overlap-y|--tile-max-size)
      option=$1
      if [[ $# -lt 2 || -z "$2" ]]; then printf '错误：%s 需要一个值。\n' "$option" >&2; exit 2; fi
      case "$option" in
        --page) page=$2 ;;
        --output) output=$2 ;;
        --config) config=$2 ;;
        --dpi) dpi=$2 ;;
        --cols) cols=$2 ;;
        --rows) rows=$2 ;;
        --overlap-x) overlap_x=$2 ;;
        --overlap-y) overlap_y=$2 ;;
        --tile-max-size) tile_max=$2 ;;
      esac
      shift 2 ;;
    --no-ocr) no_ocr=1; shift ;;
    --resume) resume=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf '错误：未知参数 %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! -f "$source_file" ]]; then printf '错误：输入文件不存在：%s\n' "$source_file" >&2; exit 2; fi
if ! command -v python3 >/dev/null 2>&1; then printf '错误：当前环境中找不到 python3。\n' >&2; exit 2; fi
if ! python3 -c 'import PIL, pymupdf, httpx, pydantic, cv2' >/dev/null 2>&1; then
  printf '错误：当前 python3 缺少依赖。请激活项目环境，或执行 python3 -m pip install -r "%s/requirements.txt"。\n' "$project_dir" >&2
  exit 2
fi
if [[ "$resume" -eq 1 && -z "$output" ]]; then printf '错误：--resume 必须指定原来的 --output。\n' >&2; exit 2; fi
if [[ -z "$output" ]]; then output="$project_dir/runs/test-$(date +%Y%m%d-%H%M%S)-$$"; fi
case "$output" in /*) ;; *) output="$caller_dir/$output" ;; esac
case "$config" in /*) ;; *) config="$caller_dir/$config" ;; esac

# 计算与 prepare_image 相同的图片边界，并选择不超过目标边长的重叠网格。
grid=$(python3 - "$source_file" "$page" "$dpi" "$tile_max" "$overlap_x" "$overlap_y" "$cols" "$rows" <<'PY'
import math, sys
from pathlib import Path
import pymupdf
from PIL import Image

source, page_arg, dpi_arg, max_arg, x_arg, y_arg, cols_arg, rows_arg = sys.argv[1:]
try:
    page, dpi, max_size = int(page_arg), int(dpi_arg), int(max_arg)
    x_overlap, y_overlap = float(x_arg), float(y_arg)
    if page < 1 or dpi < 1 or max_size < 1 or not (0 <= x_overlap < 100 and 0 <= y_overlap < 100):
        raise ValueError('页码、DPI 和切片尺寸须为正数，重叠比例须在 0 到 100 之间。')
    sizes = []
    if Path(source).suffix.lower() == '.pdf':
        with pymupdf.open(source) as doc:
            if page > len(doc):
                raise ValueError(f'PDF 共有 {len(doc)} 页，指定页码为 {page}。')
            pg = doc[page-1]
            for info in pg.get_image_info(xrefs=True):
                rect = pymupdf.Rect(info['bbox']) * pg.rotation_matrix
                rect &= pg.rect
                if not rect.is_empty and not rect.is_infinite:
                    bounds = (rect * pymupdf.Matrix(dpi/72, dpi/72)).irect
                    if bounds.width > 0 and bounds.height > 0:
                        sizes.append((bounds.width, bounds.height))
        if not sizes:
            raise ValueError('PDF 指定页没有嵌入图片；矢量页面请先转换为图片。')
    else:
        with Image.open(source) as image:
            sizes.append(image.size)
    # 留出余量，兼顾模型识别效果和最大切片限制。
    def choose(length, overlap, target):
        target = min(max_size, target)
        if length <= max_size:
            return 1
        count = 1
        while math.ceil(length / (count - (count-1)*overlap/100)) > target:
            count += 1
        return count
    width, height = max(x for x, _ in sizes), max(y for _, y in sizes)
    cols = int(cols_arg) if cols_arg else choose(width, x_overlap, 1800)
    rows = int(rows_arg) if rows_arg else choose(height, y_overlap, 1700)
    for w, h in sizes:
        if not (1 <= cols <= w and 1 <= rows <= h):
            raise ValueError('切片数须为正整数，且不能超过每张图片的相应像素数。')
        tw = w if cols == 1 else math.ceil(w / (cols - (cols-1)*x_overlap/100))
        th = h if rows == 1 else math.ceil(h / (rows - (rows-1)*y_overlap/100))
        if max(tw, th) > max_size:
            raise ValueError(f'切片 {tw}×{th} 超过限制 {max_size}，请增加 --cols 或 --rows。')
    print(cols, rows)
except (OSError, ValueError, RuntimeError) as exc:
    print(f'错误：{exc}', file=sys.stderr)
    raise SystemExit(2)
PY
)
read -r cols rows <<< "$grid"
command=(python3 "$project_dir/parse_flowchart.py" "$source_file"
  --page "$page" --dpi "$dpi" --output "$output" --config "$config"
  --locate-nodes --cols "$cols" --rows "$rows"
  --overlap-x "$overlap_x" --overlap-y "$overlap_y" --tile-max-size "$tile_max" --with-edges)
if [[ "$no_ocr" -eq 1 ]]; then command+=(--no-ocr); fi
if [[ "$resume" -eq 1 ]]; then command+=(--resume); fi
printf '输入：%s\n输出：%s\n切片：%s 列 × %s 行，重叠 %s%% × %s%%\n' "$source_file" "$output" "$cols" "$rows" "$overlap_x" "$overlap_y"
if [[ "$dry_run" -eq 1 ]]; then
  python3 -c 'import shlex, sys; print("命令：" + shlex.join(sys.argv[1:]))' "${command[@]}"
  exit 0
fi
"${command[@]}"

# 按实际输出目录生成审核图；续跑时不覆盖已有审核图。
for image_dir in "$output"/image_*; do
  [[ -d "$image_dir" && -f "$image_dir/edge_run_manifest.json" ]] || continue
  edge_name=$(python3 - "$image_dir/edge_run_manifest.json" <<'PY'
import json, sys
from pathlib import Path
name = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['output']
if not isinstance(name, str) or Path(name).name != name or not name.startswith('edges'):
    raise SystemExit('错误：边结果目录记录无效。')
print(name)
PY
)
  edge_dir="$image_dir/$edge_name"
  review="$edge_dir/review"
  suffix=1
  while [[ -e "$review" ]]; do
    review="$edge_dir/review_$suffix"
    suffix=$((suffix + 1))
  done
  render=(python3 "$project_dir/visualize_graph.py" "$image_dir/image.png" "$edge_dir/graph.json"
    --evidence "$edge_dir/edge_evidence.json" --output "$review")
  if [[ -f "$edge_dir/box_refinements.json" ]]; then render+=(--box-audit "$edge_dir/box_refinements.json"); fi
  "${render[@]}"
  printf '结果 JSON：%s\n审核图：%s\n' "$edge_dir/graph.json" "$review/global_marked.png"
done
printf '用量统计：%s\n' "$output/model_usage_summary.json"
