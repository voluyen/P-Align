"""
Xem vì sao bước lọc Eq.9 loại nhiều mẫu đến vậy.

Với mỗi mẫu bị loại, in ra đáp án chuẩn và đáp án model sinh, kèm phán quyết
của từng tầng trong bộ chấm. Mục đích là phân biệt hai nguyên nhân rất khác
nhau nhưng nhìn con số retention thì giống hệt:

  - model thực sự làm sai      -> đành chịu, đó là năng lực của student
  - bộ chấm so sai             -> lỗi của ta, sửa được

    python src/diagnose_filter.py --input data/processed/aligned.jsonl -n 25
"""

import argparse
import json
import collections

from build_align_dataset import extract_boxed, _math_verify_match, _oat_match

# Đo từ qizheyanger/P-ALIGN (P-ALIGN.json, 966 dòng), đơn vị ký tự.
# Dùng làm mốc: prefix của ta lệch nhiều so với đây thì vấn đề nằm ở giai đoạn 1,
# không phải ở năng lực của student.
AUTHOR = {
    "prefix":       {"p10": 875,  "p50": 1538, "p90": 5450,  "mean": 2795},
    "continuation": {"p10": 2612, "p50": 4791, "p90": 9550,  "mean": 5567},
    "ratio":        {"p10": 0.115, "p50": 0.254, "p90": 0.564},
}


def pct(v, p):
    v = sorted(v)
    return v[min(int(len(v) * p), len(v) - 1)] if v else 0


def compare_lengths(data):
    pre = [len((d.get("sufficient_reasoning") or "")) for d in data]
    con = [len((d.get("output") or "")) for d in data]
    rat = [p / (p + c) for p, c in zip(pre, con) if p + c]

    print("\n" + "=" * 66)
    print("  ĐỘ DÀI (ký tự) — của bạn so với dữ liệu tác giả")
    print(f"  {'':14} {'p10':>10} {'p50':>10} {'p90':>10}")
    for name, mine in (("prefix", pre), ("continuation", con)):
        a = AUTHOR[name]
        print(f"  {name:14} {pct(mine,.10):>10} {pct(mine,.50):>10} {pct(mine,.90):>10}   <- bạn")
        print(f"  {'':14} {a['p10']:>10} {a['p50']:>10} {a['p90']:>10}   <- tác giả")
    a = AUTHOR["ratio"]
    print(f"  {'tỉ lệ prefix':14} {pct(rat,.10):>10.3f} {pct(rat,.50):>10.3f} {pct(rat,.90):>10.3f}   <- bạn")
    print(f"  {'':14} {a['p10']:>10.3f} {a['p50']:>10.3f} {a['p90']:>10.3f}   <- tác giả")


def stage1_stats(path):
    try:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except OSError:
        return
    ok = [r for r in rows if r.get("is_sufficient")]
    ratios = [r["prefix_ratio"] for r in rows if "prefix_ratio" in r]
    print("\n" + "=" * 66)
    print(f"  GIAI ĐOẠN 1 ({path})")
    print(f"  is_sufficient=True : {len(ok)}/{len(rows)}"
          f"  ({len(ok)/max(len(rows),1)*100:.1f}%)")
    if ratios:
        print(f"  prefix_ratio       : p10={pct(ratios,.10):.3f} "
              f"p50={pct(ratios,.50):.3f} p90={pct(ratios,.90):.3f}")
    print("  prefix_ratio dồn về 1.0 nghĩa là student luôn trả [NOT_ENOUGH],")
    print("  hoặc nhị phân rơi vào nhánh fallback -> prefix vô dụng.")


def tiers(pred_text, gold):
    """Chạy riêng từng tầng của is_correct() để biết tầng nào gật, tầng nào lắc."""
    out = {}
    try:
        out["math_verify"] = _math_verify_match(pred_text, gold)
    except Exception as e:
        out["math_verify"] = f"ERR:{type(e).__name__}"
    out["oat"] = _oat_match(pred_text, gold)
    boxed = extract_boxed(pred_text)
    out["string"] = (boxed is not None
                     and boxed.strip().replace(" ", "") == gold.replace(" ", ""))
    return out, boxed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="output của prefix-alignment.py")
    p.add_argument("-n", type=int, default=25, help="in bao nhiêu mẫu bị loại")
    p.add_argument("--truncated", default="data/processed/truncated.jsonl",
                   help="output giai đoạn 1, để xem prefix_ratio")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]

    shown = 0
    stats = collections.Counter()
    gold_shape = collections.Counter()

    for item in data:
        gold = str(item.get("answer", "")).strip()
        cont = (item.get("output") or "").rstrip()
        if not gold or not cont:
            stats["không có gold hoặc output"] += 1
            continue

        t, boxed = tiers(cont, gold)
        if any(v is True for v in t.values()):
            stats["ĐẠT"] += 1
            continue

        stats["BỊ LOẠI"] += 1
        # phân loại hình dạng của gold: đây là chỗ dễ sai nhất
        if "=" in gold:
            gold_shape["gold chứa '=' (vd 'N = 21')"] += 1
        elif len(gold) > 40:
            gold_shape["gold dài > 40 ký tự"] += 1
        elif any(c.isalpha() for c in gold):
            gold_shape["gold có chữ cái"] += 1
        else:
            gold_shape["gold thuần số/ký hiệu"] += 1

        if shown < args.n:
            shown += 1
            print(f"\n--- id={item.get('id')} ---")
            print(f"  gold        : {gold[:90]!r}")
            print(f"  model boxed : {(boxed or '(không có)')[:90]!r}")
            print(f"  tầng        : {t}")

    print("\n" + "=" * 60)
    for k, v in stats.most_common():
        print(f"  {k:28} {v}")
    print("\n  hình dạng gold của các mẫu BỊ LOẠI:")
    for k, v in gold_shape.most_common():
        print(f"    {k:34} {v}")
    print("\n  Nếu 'gold chứa =' hoặc 'gold dài' chiếm phần lớn -> lỗi ở khâu trích")
    print("  đáp án từ cột solution, không phải model làm sai.")

    compare_lengths(data)
    stage1_stats(args.truncated)


if __name__ == "__main__":
    main()
