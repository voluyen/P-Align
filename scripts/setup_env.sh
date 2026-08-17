#!/bin/bash
# Dựng môi trường P-ALIGN trên server linux-64 + NVIDIA GPU.
#   bash scripts/setup_env.sh
# Kịch bản này KHÔNG tự chạy conda init; hãy activate env trước khi gọi.

set -euo pipefail

ENV_NAME="${ENV_NAME:-P-ALIGN}"

echo "=== Kiểm tra tiền đề ==="
command -v nvidia-smi >/dev/null || { echo "❌ Không thấy nvidia-smi."; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
    echo
    echo "❌ Chưa activate env '${ENV_NAME}'. Chạy trước:"
    echo "     conda create --name ${ENV_NAME} --file requirements.txt"
    echo "     conda activate ${ENV_NAME}"
    echo "   (requirements.txt là conda export cho linux-64, không phải file pip)"
    exit 1
fi

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available())"

echo
echo "=== Cài các gói THIẾU trong requirements.txt ==="
# requirements.txt không có 4 gói này nhưng pipeline cần:
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
echo "=== (tuỳ chọn) flash-attn: tiết kiệm nhiều VRAM ở seq 8k+ ==="
echo "   pip install flash-attn --no-build-isolation     # biên dịch lâu 10-30 phút"
echo "   rồi đổi --attn-impl flash_attention_2 trong scripts/train_sft.sh"

echo
echo "=== Kiểm tra lại ==="
python - <<'PY'
mods = ["torch", "transformers", "vllm", "deepspeed", "peft",
        "pyarrow", "huggingface_hub", "jsonlines", "math_verify", "tqdm"]
for m in mods:
    try:
        __import__(m)
        print(f"  ok       {m}")
    except Exception as e:
        print(f"  THIẾU    {m}  ({type(e).__name__})")
try:
    __import__("oat_math_grader"); print("  ok       oat_math_grader")
except Exception:
    print("  THIẾU    oat_math_grader  (tuỳ chọn cho giai đoạn 0-3, BẮT BUỘC cho giai đoạn 4)")
PY

mkdir -p output/log data/processed
echo
echo "✅ Xong. Bước tiếp: LIMIT=20 bash scripts/Prepare_data.sh"
