"""
构建 D_align：正确性过滤 + 监督信号拼接。

对应论文 Sec. 3.2.2：
  - Eq. 9  D_align = {(q^i, R~^i (+) y^i) | Ans(y^i) = a_*^i}
  - Table 10/11  监督信号形如 <Begin_of_prefix>...<End_of_prefix> + continuation

输入 = src/prefix-alignment.py 的输出 jsonl
      {id, question, answer, sufficient_reasoning, output}
输出 = SFT 训练数据 jsonl
      {id, question, answer, prompt, target, prefix, continuation}

训练时对 prompt 做 label mask，loss 只算在 target 上（论文 Eq. 2）。
"""

import argparse
import json
import os
import signal
from tqdm import tqdm

# 论文 Figure 5：Instruct_QA，直接问答模板。
# 训练与推理必须使用同一个模板，否则会引入 train/test mismatch。
INSTRUCT_QA = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "*Problem*: {problem}"
)

BEGIN_PREFIX = "<Begin_of_prefix>"
END_PREFIX = "<End_of_prefix>"


# =====================
# 答案抽取 Ans(.)
# =====================
def extract_boxed(text: str):
    """
    取最后一个 \\boxed{...} 的内容，按花括号配对解析，
    可正确处理嵌套与 LaTeX 转义（如 \\boxed{\\{1,2\\}}）。

    Returns:
        str | None: 抽到的答案，找不到时为 None。
    """
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None

    i = idx + len("\\boxed")
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None

    depth = 0
    start = i + 1
    while i < len(text):
        c = text[i]
        if c == "\\":          # 跳过被转义的字符，避免把 \{ \} 计入配对
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


# =====================
# 正确性判定
# =====================
def timeout(seconds: int = 10):
    """给 sympy 验证加超时，避免个别样本卡死（仅 POSIX）。"""
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError("Verification timed out.")

        def wrapper(*args, **kwargs):
            if os.name != "posix":
                return func(*args, **kwargs)
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


@timeout(seconds=10)
def _math_verify_match(pred_text: str, gold: str) -> bool:
    from math_verify import parse, verify
    return bool(verify(parse("$" + gold + "$"), parse(pred_text)))


def _oat_match(pred_text: str, gold: str) -> bool:
    try:
        from oat_math_grader import boxed_reward_fn
        _, r = boxed_reward_fn(pred_text, gold, fast=False)
        return r == 1.0
    except Exception:
        return False


def is_correct(pred_text: str, gold: str, use_oat: bool = True) -> bool:
    """
    判定生成结果 y 的最终答案是否等于 ground truth。
    与 src/evaluation.py 保持同一套判定口径：math_verify ∨ OAT ∨ 归一化字符串比较。
    """
    gold = str(gold).strip()
    if not gold:
        return False

    try:
        if _math_verify_match(pred_text, gold):
            return True
    except Exception:
        pass

    if use_oat and _oat_match(pred_text, gold):
        return True

    boxed = extract_boxed(pred_text)
    if boxed is not None:
        return boxed.strip().replace(" ", "") == gold.replace(" ", "")
    return False


# =====================
# 主流程
# =====================
def build(input_file: str,
          output_file: str,
          use_oat: bool = True,
          do_filter: bool = True):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"🔍 读入 {len(data)} 条 (来自 {input_file})")

    kept, n_no_output, n_no_boxed, n_wrong, n_no_answer = [], 0, 0, 0, 0

    for idx, item in enumerate(tqdm(data, desc="构建 D_align")):
        prefix = (item.get("sufficient_reasoning") or "").strip()
        continuation = (item.get("output") or "").strip()
        answer = str(item.get("answer", "")).strip()

        if not prefix or not continuation:
            n_no_output += 1
            continue

        if extract_boxed(continuation) is None:
            n_no_boxed += 1

        if do_filter:
            if "answer" not in item:
                # 字段整个缺失 = 上游接错，直接报错，别静默过滤成空集。
                raise ValueError(
                    f"样本 {item.get('id', idx)} 没有 answer 字段，无法做 Eq.9 过滤。"
                    " 请确认 prefix-alignment.py 已透传 answer（旧版本会丢弃它）。"
                )
            if not answer:
                # 字段在但为空 = 证明题之类抽不出答案，正常丢弃。
                n_no_answer += 1
                continue
            if not is_correct(continuation, answer, use_oat=use_oat):
                n_wrong += 1
                continue

        kept.append({
            "id": item.get("id", idx),
            "question": item["question"],
            "answer": answer,
            "prompt": INSTRUCT_QA.format(problem=item["question"]),
            "target": f"{BEGIN_PREFIX}{prefix}{END_PREFIX}{continuation}",
            "prefix": prefix,
            "continuation": continuation,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(data)
    print("\n" + "-" * 60)
    print(f"输入样本数        : {total}")
    print(f"丢弃 (缺 prefix/output): {n_no_output}")
    print(f"丢弃 (answer 为空) : {n_no_answer}")
    print(f"丢弃 (答案不匹配) : {n_wrong}")
    print(f"最终保留          : {len(kept)}")
    if total:
        print(f"保留率            : {len(kept) / total:.4f}   (论文 Table 7: 0.972 / 0.988)")
    print(f"⚠️ 无 \\boxed{{}} 的生成: {n_no_boxed}")
    print(f"✅ 已写入 {output_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="prefix-alignment.py 的输出 jsonl")
    p.add_argument("--output", required=True, help="D_align 输出 jsonl")
    p.add_argument("--no-oat", action="store_true", help="关闭 OAT 兜底判分")
    p.add_argument("--no-filter", action="store_true",
                   help="跳过 Eq.9 过滤（消融用，保留全部样本）")
    args = p.parse_args()

    build(args.input, args.output,
          use_oat=not args.no_oat,
          do_filter=not args.no_filter)


if __name__ == "__main__":
    main()
