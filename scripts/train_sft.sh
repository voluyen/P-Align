#!/bin/bash
# Bước 4: SFT trên D_align (论文 Eq. 2).
# Mặc định: LoRA + ZeRO-2, vừa cho 1x A100 40GB.
#
#   bash scripts/train_sft.sh [MASTER_PORT] [GPU_NUM]

set -e

MASTER_PORT=${1:-29330}
GPU_NUM=${2:-1}
BASE_PATH=$(cd "$(dirname "$0")/.." && pwd)

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"  # phải trùng model dùng ở Prepare_data.sh
DATA_PATH="${BASE_PATH}/data/processed/d_align.jsonl"
SAVE_PATH="${BASE_PATH}/output/ckpt/p-align"

GPU_ID="${GPU_ID:-0}"                      # GPU_ID=1 bash scripts/train_sft.sh
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p output/log "${SAVE_PATH}"

# GPU phải trống, nếu không sẽ OOM sau khi đã nạp xong dữ liệu.
FREE_MB=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free \
          --format=csv,noheader,nounits 2>/dev/null || echo 0)
if [[ "${FREE_MB}" -lt "${MIN_FREE_MB:-14000}" ]]; then
    echo "❌ GPU ${GPU_ID} chỉ còn ${FREE_MB} MiB trống. Tiến trình đang giữ VRAM:"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/     /'
    exit 1
fi
echo "GPU ${GPU_ID}: ${FREE_MB} MiB trống"

# ---- LoRA (mặc định, bám sát paper: lr 5e-5, 3 epochs) ----
DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_zero2.json"
LORA_OPTS="--use-lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.05"
LR=5e-5

# ---- Full fine-tune: bỏ comment 3 dòng dưới ----
# CẢNH BÁO: 7B full FT cần ~112GB model state -> phải offload hết sang CPU
# (yêu cầu ~150GB RAM) và chậm hơn LoRA khoảng 10-20 lần trên 1x40GB.
# DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_zero3_offload.json"
# LORA_OPTS=""
# LR=1e-5

torchrun \
    --nproc_per_node ${GPU_NUM} \
    --nnodes 1 \
    --master_port ${MASTER_PORT} \
    "${BASE_PATH}/src/finetune.py" \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --save "${SAVE_PATH}" \
    --lr ${LR} \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --max-length 8192 \
    --max-prompt-length 1024 \
    --gradient-checkpointing \
    --attn-impl sdpa \
    ${LORA_OPTS} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --warmup-ratio 0.03 \
    --log-interval 10 \
    --seed 10 \
    --deepspeed \
    --deepspeed_config "${DS_CONFIG}" \
    2>&1 | tee output/log/train_sft.log
