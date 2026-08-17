"""
构建 D_align：正确性过滤 + 监督信号拼接。

对应论文 Sec. 3.2.2：
  - Eq. 9  D_align = {(q^i, R~^i (+) y^i) | Ans(y^i) = a_*^i}
  - Table 10/11  监督信号 = <Begin_of_Prefix>前缀<End_of_Prefix>续写

输入 = src/prefix-alignment.py 的输出 jsonl
      {id, question, answer, sufficient_reasoning, output}
输出 = alpaca 格式 jsonl，字段与作者公开的 qizheyanger/P-ALIGN 对齐
      {id, answer, instruction, input, output}
      其中 id/answer 是多出来的排查信息，LLaMA-Factory 会忽略。

训练时 instruction+input 属于 prompt，做 label mask；loss 只算在 output 上
（论文 Eq. 2）。
"""

import argparse
import json
import os
import signal
from tqdm import tqdm

# 下面三个常量都按作者公开的 qizheyanger/P-ALIGN（P-ALIGN.json，966 条）核对过：
#   - instruction 全库只有这一种写法，问题单独放在 input 字段
#   - marker 的 P 是大写。论文 Table 11 的排版看着像小写，实际数据是大写
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

BEGIN_PREFIX = "<Begin_of_Prefix>"
END_PREFIX = "<End_of_Prefix>"


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
          do_filter: bool = True,
          strict_eq9: bool = False):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"🔍 读入 {len(data)} 条 (来自 {input_file})")

    kept, n_no_output, n_no_boxed, n_wrong, n_no_answer, n_unverified = [], 0, 0, 0, 0, 0

    for idx, item in enumerate(tqdm(data, desc="构建 D_align")):
        prefix = (item.get("sufficient_reasoning") or "").strip()
        # 只切尾部空白：作者的数据里 <End_of_Prefix> 后面紧跟着模型生成的换行，
        # 用 strip() 会把它抹掉，输出就和 qizheyanger/P-ALIGN 差一个字符。
        continuation = (item.get("output") or "").rstrip()
        answer = str(item.get("answer", "")).strip()

        if not prefix or not continuation:
            n_no_output += 1
            continue

        has_boxed = extract_boxed(continuation) is not None
        if not has_boxed:
            n_no_boxed += 1

        if do_filter:
            if "answer" not in item:
                # 字段整个缺失 = 上游接错，直接报错，别静默过滤成空集。
                raise ValueError(
                    f"样本 {item.get('id', idx)} 没有 answer 字段，无法做 Eq.9 过滤。"
                    " 请确认 prefix-alignment.py 已透传 answer（旧版本会丢弃它）。"
                )
            if not answer:
                # 证明题：没有可比对的最终答案。作者公开的 966 条里这类题全在，
                # 所以默认保留（只要 continuation 给出了 \boxed{}），
                # --strict-eq9 才按 Eq.9 的字面意思丢掉。
                if strict_eq9 or not has_boxed:
                    n_no_answer += 1
                    continue
                n_unverified += 1
            elif not is_correct(continuation, answer, use_oat=use_oat):
                n_wrong += 1
                continue

        # 字段名与 qizheyanger/P-ALIGN 一致（LLaMA-Factory 的 alpaca 格式）。
        # id/answer 是额外信息，LLaMA-Factory 会忽略，方便自己排查。
        kept.append({
            "id": item.get("id", idx),
            "answer": answer,
            "instruction": INSTRUCTION,
            "input": item["question"],
            "output": f"{BEGIN_PREFIX}{prefix}{END_PREFIX}{continuation}",
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(data)
    print("\n" + "-" * 60)
    print(f"输入样本数        : {total}")
    print(f"丢弃 (缺 prefix/output): {n_no_output}")
    print(f"丢弃 (无法比对)   : {n_no_answer}")
    print(f"保留但未验证正确性: {n_unverified}   (证明题，只检查了有 \\boxed{{}})")
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
    p.add_argument("--strict-eq9", action="store_true",
                   help="按 Eq.9 字面意思，丢掉所有没有可比对答案的证明题。默认保留，与作者公开数据一致。")
    args = p.parse_args()

    build(args.input, args.output,
          use_oat=not args.no_oat,
          do_filter=not args.no_filter,
          strict_eq9=args.strict_eq9)


if __name__ == "__main__":
    main()
