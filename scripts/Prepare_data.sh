#!/bin/bash
# =============================================================================
# P-ALIGN — chuẩn bị dữ liệu huấn luyện (D_align), chạy 3 giai đoạn liên tiếp.
#
#   Giai đoạn 1  Adaptive prefix truncation (binary search + self-judging)
#   Giai đoạn 2  Prefix-based alignment (vLLM sinh continuation)
#   Giai đoạn 3  Lọc Ans(y)=a* + ghép supervision signal  -> D_align
#
#   Giai đoạn 0  Tải HF dataset + đổi schema về {question, Long-CoT, answer}
#
# Cách dùng:
#   bash scripts/Prepare_data.sh              # chạy hết, bỏ qua giai đoạn đã xong
#   LIMIT=20 bash scripts/Prepare_data.sh     # chạy thử 20 mẫu trước khi chạy thật
#   FORCE=1 bash scripts/Prepare_data.sh      # chạy lại từ đầu (GHI ĐÈ output cũ)
#   STAGE=2 bash scripts/Prepare_data.sh      # chỉ chạy đúng 1 giai đoạn
#
# Chạy nền:  nohup bash scripts/Prepare_data.sh > output/log/prepare.log 2>&1 &
# =============================================================================

set -euo pipefail

# ----------------------------- CẤU HÌNH --------------------------------------
HF_REPO="${HF_REPO:-VoCuc/s1K-1.1-DeepSeek-R1-Distill-Qwen-32B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"     # student model (self-judge + continuation)
GPU_ID="${GPU_ID:-0}"
WORK_DIR="${WORK_DIR:-data/processed}"
LIMIT="${LIMIT:-0}"                           # >0 = chạy thử trên N mẫu đầu

RAW_DATA="${RAW_DATA:-${WORK_DIR}/s1k_prepared.jsonl}"   # output giai đoạn 0
TRUNCATED="${WORK_DIR}/truncated.jsonl"       # output giai đoạn 1
ALIGNED="${WORK_DIR}/aligned.jsonl"           # output giai đoạn 2
D_ALIGN="${WORK_DIR}/d_align.jsonl"           # output giai đoạn 3 -> dùng để train

STAGE="${STAGE:-all}"
FORCE="${FORCE:-0}"

# conda env P-ALIGN có 'python'; một số máy chỉ có 'python3'.
PY_BIN="${PY_BIN:-$(command -v python || command -v python3 || true)}"
[[ -n "${PY_BIN}" ]] || { echo "❌ Không tìm thấy python. Đã 'conda activate P-ALIGN' chưa?"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PALIGN_MODEL="${MODEL_PATH}"
export PALIGN_SLEEP="${PALIGN_SLEEP:-0}"      # 0 = bỏ sleep vô ích của model local
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # giảm phân mảnh VRAM

# Bao nhiêu MiB trống mới đủ nạp student model (Qwen3-4B bf16 ~8GB + activation)
MIN_FREE_MB="${MIN_FREE_MB:-12000}"

mkdir -p "${WORK_DIR}" output/log

# ----------------------------- TIỆN ÍCH --------------------------------------
banner() { echo; echo "=============================================================="; \
           echo "  $*"; echo "=============================================================="; }

# Giai đoạn được coi là đã xong nếu file output tồn tại và khác rỗng.
done_already() {
    [[ "${FORCE}" != "1" && -s "$1" ]]
}

want_stage() {
    [[ "${STAGE}" == "all" || "${STAGE}" == "$1" ]]
}

lines_of() { [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0; }

# Giai đoạn 1 và 2 đều chạy tiếp được từ file dở, nên chỉ coi là xong khi số
# dòng ra đã bằng số dòng vào. Nếu chỉ kiểm tra "khác rỗng" thì một lần crash
# giữa chừng sẽ bị hiểu nhầm là đã hoàn tất, và pipeline đi tiếp với dữ liệu thiếu.
resumable_stage() {           # $1 = file ra, $2 = file vào, $3 = tên giai đoạn
    local n_out n_in
    n_out=$(lines_of "$1"); n_in=$(lines_of "$2")
    if [[ "${FORCE}" == "1" ]]; then
        rm -f "$1"; return 1
    fi
    if [[ "${n_out}" -gt 0 && "${n_out}" -ge "${n_in}" ]]; then
        echo; echo "⏭  Bỏ qua giai đoạn $3: ${n_out}/${n_in} dòng, đã xong."
        return 0
    fi
    [[ "${n_out}" -gt 0 ]] && echo "↻  Giai đoạn $3 chạy tiếp từ ${n_out}/${n_in} dòng."
    return 1
}

# GPU phải thực sự trống. Đây là lỗi hay gặp nhất: vLLM của lần chạy trước còn
# giữ ~80% VRAM, tiến trình mới chỉ xin được vài GB rồi OOM.
require_free_gpu() {
    local free_mb
    free_mb=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free \
              --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [[ "${free_mb}" -lt "${MIN_FREE_MB}" ]]; then
        echo "❌ GPU ${GPU_ID} chỉ còn ${free_mb} MiB trống (cần ≥ ${MIN_FREE_MB})."
        echo "   Tiến trình đang giữ VRAM:"
        nvidia-smi --query-compute-apps=pid,used_memory,process_name \
                   --format=csv 2>/dev/null | sed 's/^/     /'
        echo "   Kill tiến trình cũ rồi chạy lại, hoặc đổi GPU_ID=<id khác>."
        exit 1
    fi
    echo "   GPU ${GPU_ID}: ${free_mb} MiB trống"
}

# prefix-alignment.py bắt lỗi theo batch rồi continue, nên nó vẫn thoát mã 0
# dù không ghi được dòng nào. Không kiểm tra ở đây thì lỗi chỉ lộ ra ở giai
# đoạn sau dưới dạng "thiếu file", che mất nguyên nhân thật.
must_have_output() {          # $1 = file, $2 = tên giai đoạn, $3 = log
    if [[ ! -s "$1" ]]; then
        echo
        echo "❌ Giai đoạn $2 kết thúc nhưng $1 rỗng."
        echo "   30 dòng cuối của $3:"
        tail -30 "$3" 2>/dev/null | sed 's/^/     /'
        exit 1
    fi
}

# --------------------------- GIAI ĐOẠN 0 -------------------------------------
if want_stage 0; then
    if done_already "${RAW_DATA}"; then
        echo
        echo "⏭  Bỏ qua giai đoạn 0: đã có ${RAW_DATA} ($(wc -l < "${RAW_DATA}") dòng)."
    else
        banner "Giai đoạn 0/3 — Tải ${HF_REPO} và đổi schema"
        LIMIT_OPT=""
        [[ "${LIMIT}" != "0" ]] && LIMIT_OPT="--limit ${LIMIT}"
        "${PY_BIN}" src/prepare_s1k.py \
            --repo "${HF_REPO}" \
            --output "${RAW_DATA}" \
            ${LIMIT_OPT} 2>&1 | tee output/log/prepare_s1k.log
    fi
fi

# ----------------------- KIỂM TRA ĐẦU VÀO (preflight) ------------------------
# Chạy trước khi nạp model, để lỗi schema lộ ra ngay thay vì sau nhiều giờ GPU.
banner "Preflight: kiểm tra ${RAW_DATA}"

if [[ ! -f "${RAW_DATA}" ]]; then
    echo "❌ Không thấy file: ${RAW_DATA}"
    echo "   Chạy giai đoạn 0 trước: STAGE=0 bash $0"
    exit 1
fi

"${PY_BIN}" - "${RAW_DATA}" <<'PY'
import json, sys

path = sys.argv[1]
need = ("question", "Long-CoT")          # binary_search.py đọc đúng 2 key này
n = bad = 0
missing = {k: 0 for k in need + ("answer",)}

with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        n += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            if bad <= 3:
                print(f"  dòng {i+1}: không parse được JSON")
            continue
        for k in missing:
            if not str(row.get(k, "")).strip():
                missing[k] += 1

print(f"  tổng số dòng   : {n}")
print(f"  JSON hỏng      : {bad}")
for k, v in missing.items():
    print(f"  thiếu '{k}'    : {v}")

if n == 0 or bad:
    sys.exit("❌ File đầu vào không hợp lệ.")

for k in need:
    if missing[k] == n:
        sys.exit(
            f"❌ Không dòng nào có key '{k}'.\n"
            f"   binary_search.py đọc {need} và 'answer'.\n"
            f"   Hãy đổi tên field trong dữ liệu LRM của bạn cho khớp."
        )
if missing["answer"] == n:
    sys.exit(
        "❌ Không dòng nào có 'answer'. Thiếu nó thì giai đoạn 3 "
        "không lọc được Ans(y)=a* (Eq.9)."
    )
print("✅ Schema hợp lệ.")
PY

echo
echo "  student model : ${MODEL_PATH}"
echo "  GPU           : ${GPU_ID}"
echo "  thư mục ra    : ${WORK_DIR}"

# --------------------------- GIAI ĐOẠN 1 -------------------------------------
if want_stage 1; then
    if ! resumable_stage "${TRUNCATED}" "${RAW_DATA}" "1"; then
        banner "Giai đoạn 1/3 — Prefix truncation (chậm nhất, tính bằng giờ)"
        require_free_gpu
        PALIGN_INPUT="${RAW_DATA}" PALIGN_OUTPUT="${TRUNCATED}" \
            "${PY_BIN}" src/binary_search.py 2>&1 | tee -a output/log/prefix.log
        must_have_output "${TRUNCATED}" "1" "output/log/prefix.log"
        echo "→ ${TRUNCATED}: $(lines_of "${TRUNCATED}") dòng"
    fi
fi

# --------------------------- GIAI ĐOẠN 2 -------------------------------------
if want_stage 2; then
    if ! resumable_stage "${ALIGNED}" "${TRUNCATED}" "2"; then
        banner "Giai đoạn 2/3 — Prefix alignment (vLLM)"
        [[ -f "${TRUNCATED}" ]] || { echo "❌ Chưa có ${TRUNCATED}. Chạy giai đoạn 1 trước."; exit 1; }
        [[ -s "${TRUNCATED}" ]] || { echo "❌ ${TRUNCATED} tồn tại nhưng RỖNG — giai đoạn 1 đã chạy mà không ra kết quả."; exit 1; }
        require_free_gpu
        # Script này tự resume theo 'question' nên an toàn khi chạy lại trên file dở.
        PALIGN_INPUT="${TRUNCATED}" PALIGN_OUTPUT="${ALIGNED}" \
            "${PY_BIN}" src/prefix-alignment.py 2>&1 | tee -a output/log/prefix-alignment.log
        must_have_output "${ALIGNED}" "2" "output/log/prefix-alignment.log"
        echo "→ ${ALIGNED}: $(lines_of "${ALIGNED}") dòng"
    fi
fi

# --------------------------- GIAI ĐOẠN 3 -------------------------------------
if want_stage 3; then
    banner "Giai đoạn 3/3 — Lọc Eq.9 + ghép supervision signal"
    [[ -f "${ALIGNED}" ]] || { echo "❌ Chưa có ${ALIGNED}. Chạy giai đoạn 2 trước."; exit 1; }
    [[ -s "${ALIGNED}" ]] || { echo "❌ ${ALIGNED} tồn tại nhưng RỖNG — giai đoạn 2 đã chạy mà mọi batch đều lỗi.
   Xem: tail -40 output/log/prefix-alignment.log"; exit 1; }
    "${PY_BIN}" src/build_align_dataset.py \
        --input "${ALIGNED}" \
        --output "${D_ALIGN}" \
        2>&1 | tee output/log/build_align.log
fi

banner "XONG"
if [[ -s "${D_ALIGN}" ]]; then
    echo "Dữ liệu huấn luyện: ${D_ALIGN}  ($(wc -l < "${D_ALIGN}") dòng)"
    echo
    echo "Bước tiếp theo:  bash scripts/train_sft.sh"
else
    echo "Chưa có ${D_ALIGN} — còn giai đoạn chưa chạy."
    echo "Chạy đủ 3 giai đoạn:  bash scripts/Prepare_data.sh"
fi
