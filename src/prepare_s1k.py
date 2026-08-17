"""Giai đoạn 0: HF dataset -> jsonl mà src/binary_search.py đọc được.

Nguồn mặc định: VoCuc/s1K-1.1-DeepSeek-R1-Distill-Qwen-32B

    global_idx          -> id
    question            -> question
    generated_response  -> Long-CoT
    solution            -> answer        (cần trích, xem bên dưới)

Cột top_token_ids / top_logprobs bị bỏ: chúng chiếm gần hết 7.05 GB của repo,
mà P-ALIGN không dùng logit teacher (Eq. 2 là NLL thuần, không có KD loss).
Script đọc thẳng parquet trên Hub và chỉ kéo 4 cột cần thiết, nên chỉ tải vài
chục MB. Dùng `datasets.load_dataset` sẽ tải đủ 7 GB rồi mới bỏ cột.

Về cột `solution`: nó KHÔNG phải đáp án cuối mà là lời giải đầy đủ (trung bình
~1000 ký tự). Có ba dạng:
    - đáp án số ngắn        "128", "109", "167.0"      -> dùng nguyên
    - lời giải dài kết \\boxed{...}                     -> trích trong hộp
    - bài chứng minh kết \\blacksquare                  -> KHÔNG có đáp án cuối
Dạng thứ ba không thể lọc bằng Eq.9 (Ans(y)=a* không xác định) nên mặc định bị
loại. Dùng --keep-unextractable nếu bạn muốn giữ và tự xử lý.
"""

import argparse
import json
import os

from build_align_dataset import extract_boxed

COLUMNS = ["global_idx", "question", "generated_response", "solution"]


def load_rows(repo, split, revision=None, limit=0):
    """Đọc parquet trên Hub với column projection.

    pyarrow chỉ tải các column chunk được chọn qua HTTP range request, nên
    top_token_ids / top_logprobs không bao giờ chạm đĩa.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    kw = {"revision": revision} if revision else {}
    files = sorted(fs.glob(f"datasets/{repo}/**/{split}-*.parquet", **kw))
    if not files:
        files = sorted(fs.glob(f"datasets/{repo}/**/*.parquet", **kw))
    if not files:
        raise FileNotFoundError(
            f"Không thấy file parquet nào trong datasets/{repo}. "
            "Repo có phải dataset public không?"
        )

    print(f"   {len(files)} shard parquet, chỉ đọc {COLUMNS}")
    rows = []
    for i, path in enumerate(files):
        table = pq.read_table(path, columns=COLUMNS, filesystem=fs)
        rows.extend(table.to_pylist())
        print(f"   shard {i+1}/{len(files)} -> {len(rows)} dòng", flush=True)
        if limit and len(rows) >= limit:
            break   # chạy thử thì không cần đọc hết 15 shard
    return rows[:limit] if limit else rows


def extract_answer(solution: str, max_short: int = 32):
    """Trả (answer, nguồn). nguồn ∈ {boxed, short, none}."""
    s = (solution or "").strip()
    if not s:
        return None, "none"

    boxed = extract_boxed(s)
    if boxed is not None and boxed.strip():
        return boxed.strip(), "boxed"

    # Đáp án số ngắn, một dòng: dùng nguyên văn.
    if len(s) <= max_short and "\n" not in s:
        return s.rstrip("."), "short"

    return None, "none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="VoCuc/s1K-1.1-DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--split", default="train")
    p.add_argument("--revision", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="chỉ lấy N dòng đầu, để chạy thử cả pipeline cho nhanh")
    p.add_argument("--max-short", type=int, default=32,
                   help="solution ngắn hơn ngần này (và 1 dòng) được coi là đáp án")
    p.add_argument("--keep-unextractable", action="store_true",
                   help="giữ cả dòng không trích được đáp án (answer rỗng). "
                        "Cảnh báo: giai đoạn 3 sẽ loại sạch chúng.")
    args = p.parse_args()

    print(f"📥 Đang đọc {args.repo} (split={args.split})...")
    data = load_rows(args.repo, args.split, args.revision, args.limit)
    print(f"   lấy được {len(data)} dòng")

    src_count = {"boxed": 0, "short": 0, "none": 0}
    n_no_cot = 0
    rows = []

    for i, item in enumerate(data):
        cot = (item.get("generated_response") or "").strip()
        question = (item.get("question") or "").strip()
        if not cot or not question:
            n_no_cot += 1
            continue

        answer, src = extract_answer(item.get("solution"), args.max_short)
        src_count[src] += 1
        if answer is None and not args.keep_unextractable:
            continue

        rows.append({
            "id": item.get("global_idx", i),
            "question": question,
            "Long-CoT": cot,
            "answer": answer or "",
        })

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(data)
    print("\n" + "-" * 62)
    print(f"tổng số dòng nguồn            : {total}")
    print(f"thiếu question/Long-CoT       : {n_no_cot}")
    print(f"answer lấy từ \\boxed{{}}       : {src_count['boxed']}")
    print(f"answer lấy từ solution ngắn   : {src_count['short']}")
    print(f"KHÔNG trích được (chứng minh) : {src_count['none']}"
          f"{'  (đã giữ lại)' if args.keep_unextractable else '  (đã loại)'}")
    print(f"ghi ra                        : {len(rows)} dòng -> {args.output}")

    if rows:
        lens = sorted(len(r["Long-CoT"]) for r in rows)
        print(f"độ dài Long-CoT (ký tự)       : p50={lens[len(lens)//2]}  "
              f"p95={lens[int(len(lens)*0.95)]}  max={lens[-1]}")

    if src_count["none"]:
        pct = src_count["none"] / max(total, 1) * 100
        print(f"\n⚠️  {pct:.1f}% là bài chứng minh, không có đáp án cuối để so khớp.")
        print("   Đây là đặc tính của s1K-1.1 (trộn thi đấu + chứng minh), không phải lỗi.")
        print("   Retention ở giai đoạn 3 vì thế sẽ thấp hơn 97-99% của Table 7.")

    if len(rows) == 0:
        raise SystemExit("❌ Không ghi được dòng nào.")


if __name__ == "__main__":
    main()
