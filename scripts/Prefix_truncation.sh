#!/bin/bash
# Bước 1: Adaptive prefix truncation via binary search (论文 Sec. 3.2.1)
# 输入/输出路径在 src/binary_search.py 的 __main__ 中修改。

export CUDA_VISIBLE_DEVICES="0"   # 改成你的 GPU id

mkdir -p output/log
nohup python src/binary_search.py > output/log/prefix.log 2>&1 &
echo "started: tail -f output/log/prefix.log"
