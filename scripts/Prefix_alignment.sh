#!/bin/bash
# Bước 2: Prefix-based alignment (论文 Sec. 3.2.2, Eq. 8)
# 输入/输出路径在 src/prefix-alignment.py 的 main() 中修改。

export CUDA_VISIBLE_DEVICES="0"   # 改成你的 GPU id

mkdir -p output/log
nohup python src/prefix-alignment.py > output/log/prefix-alignment.log 2>&1 &
echo "started: tail -f output/log/prefix-alignment.log"
