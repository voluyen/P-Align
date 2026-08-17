#!/bin/bash
# Bước 3: 正确性过滤 (Eq. 9) + 监督信号拼接 → D_align
# CPU-only，không cần GPU.

mkdir -p output/log

python src/build_align_dataset.py \
    --input  "data/processed/aligned.jsonl" \
    --output "data/processed/d_align.jsonl" \
    2>&1 | tee output/log/build_align.log
