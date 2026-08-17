#!/bin/bash
# Dựng môi trường P-ALIGN trên server linux-64 + NVIDIA GPU.
#
#   conda create -n P-ALIGN python=3.10 -y
#   conda activate P-ALIGN
#   bash scripts/setup_env.sh
#
# LƯU Ý về requirements.txt: đây là output của `conda list --export`, trộn hai
# loại dòng:
#   - 29 gói conda thật   (libgcc, openssl, python...) -> chỉ là lib hệ thống
#   - 145 gói cài bằng pip, đánh dấu `=pypi_0`         -> toàn bộ phần quan trọng
# Vì thế CẢ HAI cách dưới đây đều KHÔNG chạy được:
#   conda create --file requirements.txt   -> PackagesNotFoundError ở dòng pypi_0
#   pip install -r requirements.txt        -> pip không parse được `name=ver=build`
# Script này tách các dòng pypi_0 ra và đổi sang `name==version` cho pip.

set -euo pipefail

ENV_NAME="${ENV_NAME:-P-ALIGN}"
REQ_IN="requirements.txt"
REQ_PIP="output/requirements-pip.txt"

echo "=== Kiểm tra tiền đề ==="
command -v nvidia-smi >/dev/null || { echo "❌ Không thấy nvidia-smi."; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
    cat <<EOF

❌ Chưa activate env '${ENV_NAME}'. Chạy trước:

     conda create -n ${ENV_NAME} python=3.10 -y
     conda activate ${ENV_NAME}
     bash scripts/setup_env.sh

(Đừng dùng 'conda create --file requirements.txt' — nó sẽ báo
 PackagesNotFoundError vì các dòng =pypi_0 không có trên kênh conda.)
EOF
    exit 1
fi

[[ -f "${REQ_IN}" ]] || { echo "❌ Không thấy ${REQ_IN}. Chạy script từ thư mục gốc repo."; exit 1; }
mkdir -p output/log data/processed

echo
echo "=== Sinh danh sách pip từ ${REQ_IN} ==="
# Bỏ comment, giữ dòng có hậu tố =pypi_0, đổi `name=ver=pypi_0` -> `name==ver`.
grep -v '^#' "${REQ_IN}" | grep '=pypi_0$' | sed 's/=pypi_0$//' | sed 's/=/==/' > "${REQ_PIP}"
echo "   ${REQ_PIP}: $(wc -l < "${REQ_PIP}") gói"

echo
echo "=== Cài (mất 10-20 phút, riêng vllm + torch đã vài GB) ==="
# Không dùng --no-deps: danh sách này là snapshot của một env chạy được nên
# vốn đã nhất quán, để pip tự giải phụ thuộc thì an toàn hơn.
pip install --no-cache-dir -r "${REQ_PIP}"

echo
echo "=== Cài các gói THIẾU trong requirements.txt ==="
#   deepspeed / peft  -> huấn luyện (src/finetune.py)
#   oat_math_grader   -> chấm điểm (src/evaluation.py, tầng 2 của bộ lọc Eq.9)
pip install --no-cache-dir "deepspeed>=0.14" "peft>=0.11"

echo
echo "--- oat_math_grader (không có trên PyPI, lấy từ nguồn) ---"
pip install --no-cache-dir "git+https://github.com/sail-sg/oat.git#subdirectory=oat_math_grader" \
    || echo "⚠️  Cài oat_math_grader thất bại. Pipeline vẫn chạy được:
    - build_align_dataset.py import lười, tự bỏ qua tầng OAT
    - evaluation.py import ở module scope -> giai đoạn 4 sẽ lỗi cho tới khi cài được"

echo
echo "=== Kiểm tra lại ==="
python - <<'PY'
import importlib, sys
core = ["torch", "transformers", "vllm", "deepspeed", "peft",
        "pyarrow", "huggingface_hub", "jsonlines", "math_verify", "tqdm"]
missing = []
for m in core:
    try:
        importlib.import_module(m)
        print(f"  ok       {m}")
    except Exception as e:
        print(f"  THIẾU    {m}  ({type(e).__name__})")
        missing.append(m)
try:
    importlib.import_module("oat_math_grader"); print("  ok       oat_math_grader")
except Exception:
    print("  THIẾU    oat_math_grader  (tuỳ chọn cho giai đoạn 0-3, BẮT BUỘC cho giai đoạn 4)")

import torch
print(f"\n  torch {torch.__version__} | cuda {torch.version.cuda} | GPU khả dụng: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  ⚠️  torch không thấy GPU — kiểm tra driver so với bản CUDA của torch.")
if missing:
    sys.exit(f"\n❌ Còn thiếu: {', '.join(missing)}")
PY

echo
echo "=== (tuỳ chọn) flash-attn: tiết kiệm nhiều VRAM ở seq 8k+ ==="
echo "   pip install flash-attn --no-build-isolation    # biên dịch 10-30 phút"
echo "   rồi đổi --attn-impl flash_attention_2 trong scripts/train_sft.sh"
echo
echo "✅ Xong. Bước tiếp: LIMIT=20 bash scripts/Prepare_data.sh"
