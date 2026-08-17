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


if __name__ == "__main__":
    main()
