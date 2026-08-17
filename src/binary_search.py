"""
阶段一：自评判 + 二分查找，截出最短充分前缀（论文 Sec. 3.2.1）。

改用 vLLM 并把所有样本按“轮次”并排跑。这不改变算法本身，只改变调度方式：
  - 论文 A.5 说明二分轮数固定为 O(log2 n)，与判定结果无关，
    所以不同样本的第 k 轮可以合并成一次 llm.generate()。
  - 每个样本各轮的 prompt 是嵌套前缀（sentences[:6] ⊂ sentences[:12]），
    且变化的部分在 prompt 末尾，正好命中 vLLM 的 prefix caching。
原先是每个样本每轮一次 HF generate，670 条样本要串行调用约 4700 次；
现在整个数据集只需约 log2(max_m) 次批量调用。

In : {question, Long-CoT, answer, id}
Out: {id, answer, question, sufficient_reasoning, sufficient_sentences,
      total_sentences, prefix_ratio, is_sufficient, evaluator_response}
"""

import json
import os
import time

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =====================
# Sentence Splitter
# =====================
def split_sentences(text):
    """
    Simple sentence-level splitter.
    You can replace this with a more robust one if needed.
    """
    sentences = text.split(". ")
    return [
        s.strip() + "." if not s.endswith(".") else s.strip()
        for s in sentences if s.strip()
    ]


# =====================
# Sufficiency Check
# =====================
def build_eval_prompt(question, reasoning_part):
    """Instruct_Eval（论文 Figure 6）。逐字保持原样，问题在前、前缀在后。"""
    return f"""
You are a reasoning evaluator.

You are given a partial reasoning prefix extracted from a longer chain-of-thought.
Your task is to judge whether this prefix already contains the essential logical structure and key transformations needed to complete the solution.

- Reply "[ENOUGH]" if the prefix establishes the core reasoning steps such that the remaining reasoning is straightforward or routine.
- Reply "[NOT_ENOUGH]" if any crucial reasoning step is still missing, making it difficult to reliably complete the solution.

Reply with exactly one token: [ENOUGH] or [NOT_ENOUGH].

Question:
{question}

Partial reasoning:
{reasoning_part}
"""


def is_enough(response: str) -> bool:
    return "[ENOUGH]" in response or response.strip() == "ENOUGH"


def render(tokenizer, prompt):
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def max_sentences_in_context(tokenizer, question, sentences, budget):
    """
    最多能放进上下文的句子数。

    句子逐条 tokenize 后累加，是个近似值（拼接后的分词略有差异），所以留了
    5% 余量。超出上下文的样本会把二分右边界压到这里，而不是等 vLLM 抛错。
    """
    overhead = len(tokenizer(build_eval_prompt(question, ""),
                             add_special_tokens=False)["input_ids"]) + 64
    room = budget - overhead
    if room <= 0:
        return 0

    used = 0
    for i, s in enumerate(sentences):
        used += len(tokenizer(s + " ", add_special_tokens=False)["input_ids"])
        if used * 1.05 > room:
            return i
    return len(sentences)


# =====================
# Batched Binary Search
# =====================
def search_all(records, llm, tokenizer, sampling_params, max_prompt_tokens):
    """
    所有样本同步推进二分查找，每一轮合并成一次 llm.generate()。
    每轮结束后 yield 当轮刚刚搜完的样本，方便边跑边落盘。
    """
    states = []
    for rec in records:
        sents = rec["sentences"]
        right = max_sentences_in_context(
            tokenizer, rec["question"], sents, max_prompt_tokens)
        if right < len(sents):
            print(f"[Clamp] id={rec['id']} 上下文只放得下 {right}/{len(sents)} 句，"
                  f"二分右边界收缩到 {right}。")
        states.append({
            "rec": rec, "left": 1, "right": right,
            "best": None, "resp": "", "rounds": 0,
        })

    max_rounds = max((s["right"] for s in states), default=0).bit_length() + 2
    pbar = tqdm(total=max_rounds, desc="二分轮次")

    round_no = 0
    while True:
        prompts, owners = [], []
        for st in states:
            if st["best"] is not None and st["left"] > st["right"]:
                continue
            if st["left"] > st["right"] or st["right"] < 1:
                continue
            mid = (st["left"] + st["right"]) // 2
            prefix_text = " ".join(st["rec"]["sentences"][:mid])
            prompts.append(render(
                tokenizer, build_eval_prompt(st["rec"]["question"], prefix_text)))
            owners.append((st, mid))

        if not prompts:
            break

        round_no += 1
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        dt = time.time() - t0
        print(f"[Round {round_no}] {len(prompts)} 条样本，用时 {dt:.1f}s "
              f"（{dt / max(len(prompts), 1) * 1000:.0f} ms/条）")

        finished = []
        for (st, mid), out in zip(owners, outputs):
            resp = out.outputs[0].text
            st["rounds"] += 1
            if is_enough(resp):
                st["best"], st["resp"] = mid, resp
                st["right"] = mid - 1        # 往左找更短的
            else:
                st["left"] = mid + 1         # 往右加长
            if st["left"] > st["right"]:
                finished.append(st)

        pbar.update(1)
        yield finished

    pbar.close()

    # 收尾：右边界被压到 0 的样本一轮都没跑过
    leftover = [st for st in states if st["left"] > st["right"] and st["rounds"] == 0]
    if leftover:
        yield leftover


def to_result(st):
    rec, sents = st["rec"], st["rec"]["sentences"]
    if st["best"] is None:
        text, n, ok = " ".join(sents), len(sents), False
    else:
        text, n, ok = " ".join(sents[:st["best"]]), st["best"], True
    return {
        "id": rec["id"],
        "answer": rec["answer"],
        "question": rec["question"],
        "sufficient_reasoning": text,
        "sufficient_sentences": n,
        "total_sentences": len(sents),
        "prefix_ratio": n / max(len(sents), 1),
        "is_sufficient": ok,
        "evaluator_response": st["resp"],
    }


# =========================
# 主处理流程（JSONL）
# =========================
def process_jsonl(input_file, output_file, llm, tokenizer,
                  sampling_params, max_prompt_tokens):
    # 断点续跑：不清空输出文件，跳过已处理的题目。
    done = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question"])
                except Exception:
                    continue
        if done:
            print(f"[Resume] 已有 {len(done)} 条结果，将跳过这些题目。")

    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            question = data.get("question", "")
            full_reasoning = data.get("Long-CoT", "")
            if not question or not full_reasoning or question in done:
                continue
            records.append({
                "id": data.get("id", idx),
                "answer": data.get("answer", ""),
                "question": question,
                "sentences": split_sentences(full_reasoning),
            })

    if not records:
        print("没有需要处理的样本。")
        return

    n_sents = sorted(len(r["sentences"]) for r in records)
    print(f"待处理 {len(records)} 条；句子数 p50={n_sents[len(n_sents)//2]} "
          f"max={n_sents[-1]}（论文 Table 5 的平均搜索次数为 7.4，"
          f"对应约 100-170 句）")

    # 分块跑：一个样本各轮的 prompt 是嵌套前缀，只有当这批样本的 KV 都还留在
    # vLLM 的 prefix cache 里，下一轮才能命中。整份数据一起跑会把缓存冲掉。
    chunk = int(os.environ.get("PALIGN_CHUNK", 64))

    n_done = 0
    with open(output_file, "a", encoding="utf-8") as out:
        for c0 in range(0, len(records), chunk):
            batch = records[c0:c0 + chunk]
            print(f"\n=== 分块 {c0 // chunk + 1}/{(len(records) + chunk - 1) // chunk} "
                  f"（{len(batch)} 条）===")
            for finished in search_all(batch, llm, tokenizer,
                                       sampling_params, max_prompt_tokens):
                for st in finished:
                    out.write(json.dumps(to_result(st), ensure_ascii=False) + "\n")
                    n_done += 1
                out.flush()
                os.fsync(out.fileno())
                if finished:
                    print(f"[Saved] 累计写出 {n_done}/{len(records)} 条")

    ok = n_done
    print(f"✅ 本次写出 {ok} 条 -> {output_file}")


# =====================
# Entry Point
# =====================
def main():
    model_name = os.environ.get("PALIGN_MODEL", "Path to your model")
    input_file = os.environ.get("PALIGN_INPUT", "Path to your input jsonl file")
    output_file = os.environ.get("PALIGN_OUTPUT", "Path to your output jsonl file")

    max_model_len = int(os.environ.get("PALIGN_MAX_MODEL_LEN", 32768))
    gpu_util = float(os.environ.get("PALIGN_GPU_UTIL", 0.85))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 只需要 [ENOUGH] / [NOT_ENOUGH]，原来给 256 是在为一个标签解码上百步。
    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=16)
    llm = LLM(model=model_name, gpu_memory_utilization=gpu_util,
              max_model_len=max_model_len, trust_remote_code=True,
              tensor_parallel_size=1)

    process_jsonl(input_file, output_file, llm, tokenizer,
                  sampling_params, max_prompt_tokens=max_model_len - 64)
    print("\n✅ 全部数据处理完成")


if __name__ == "__main__":
    main()
