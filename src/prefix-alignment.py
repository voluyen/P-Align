import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import json
import re
import torch
import os
import csv
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def render(tokenizer, prompt):
    """套用 chat 模板。enable_thinking=False：推理由前缀提供，模型不该自己开思考块。"""
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params,
                 max_model_len=32768, reserve_output=2048):

    # --- 读取输入数据 ---
    with jsonlines.open(json_filename) as infile:
        data = []
        for idx, item in enumerate(infile):
            question = item['question']
            sufficient_reasoning = item['sufficient_reasoning']

            prompt = (
                f"Please continue from the draft and solve the problem step by step, and put your final answer within \\boxed{{}}. "
                f"I will provide you with some prior knowledge as a draft to assist you in solving the question."
                f"*Question*:{question}\n"
                f"*Prefix*:{sufficient_reasoning}"
            )

            # 必须透传 answer：下游 build_align_dataset.py 依赖它做
            # 论文 Eq.9 的正确性过滤 Ans(y^i) == a_*^i
            data.append({
                'id': item.get('id', idx),
                'question': question,
                'answer': item.get('answer', ''),
                'sufficient_reasoning': sufficient_reasoning,
                'prompt': prompt,
            })

    # --- 如果文件已存在，跳过已完成部分 ---
    existing = set()
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            for line in f:
                try:
                    existing_item = json.loads(line)
                    existing.add(existing_item['question'])
                except Exception:
                    continue
        print(f"[Resume] Found {len(existing)} existing entries. Will skip them.")

    # --- 预先套模板并按 token 长度过滤 ---
    # vLLM 只要发现 batch 里有一条 prompt 超长，就会让整个 llm.generate() 抛错。
    # batch_size=500 时，一条坏样本会连累 499 条好样本，所以必须提前剔除。
    max_prompt_len = max_model_len - reserve_output
    kept, n_too_long, longest = [], 0, 0
    for d in data:
        d['text'] = render(tokenizer, d['prompt'])
        n_tok = len(tokenizer(d['text'], add_special_tokens=False)['input_ids'])
        longest = max(longest, n_tok)
        if n_tok > max_prompt_len:
            n_too_long += 1
            continue
        kept.append(d)

    if n_too_long:
        print(f"⚠️  {n_too_long}/{len(data)} 条 prompt 超过 {max_prompt_len} token，已跳过"
              f"（最长 {longest}，max_model_len={max_model_len}，留给输出 {reserve_output}）。")
        print("   这些多半是二分查找没找到充分前缀、回退成完整推理的样本，"
              "prefix_ratio≈1.0，本来对 P-ALIGN 也没什么价值。")
    data = kept
    if not data:
        raise RuntimeError(
            f"所有 prompt 都超长（最长 {longest} token）。"
            f"请调大 max_model_len 或检查第一阶段是否几乎没做截断。")

    # --- 按 batch 生成 ---
    total_batches = (len(data) + batch_size - 1) // batch_size
    print(f"Total {len(data)} samples, batch_size={batch_size}, total_batches={total_batches}")

    n_written, n_failed, last_err = 0, 0, None

    # 使用 append 模式写入
    with open(file_name, "a", encoding="utf-8") as file:
        for batch_idx in tqdm(range(total_batches), total=total_batches, desc="Generating"):
            start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
            batch_data = [d for d in data[start:end] if d['question'] not in existing]

            if not batch_data:
                continue

            # --- 调用生成（模板在前面已套好）---
            texts = [d['text'] for d in batch_data]
            try:
                outputs = llm.generate(texts, sampling_params)
                pairs = list(zip(outputs, batch_data))
            except Exception as e:
                # 整批失败时逐条重试：一条坏样本不该拖垮同批的其余几百条。
                last_err = e
                print(f"[Error] Batch {batch_idx} failed ({e}); 改为逐条重试...")
                pairs = []
                for d in batch_data:
                    try:
                        out = llm.generate([d['text']], sampling_params)
                        pairs.append((out[0], d))
                    except Exception as e2:
                        n_failed += 1
                        last_err = e2
                print(f"[Recovered] {len(pairs)}/{len(batch_data)} 条救回。")

            # --- 保存结果 ---
            for output, item in pairs:
                item['output'] = output.outputs[0].text
                item.pop('text', None)          # 模板文本不必写进结果文件
                json_line = json.dumps(item, ensure_ascii=False)
                file.write(json_line + "\n")
                n_written += 1

            file.flush()  # 确保每个 batch 都立即落盘
            os.fsync(file.fileno())

            print(f"[Saved] Batch {batch_idx+1}/{total_batches} ({len(batch_data)} items) written.")

    # 之前这里无论如何都打印“成功”并以 0 退出。一旦每个 batch 都失败，就会留下
    # 一个空文件，问题要到下一阶段才以“缺少文件”的形式冒出来，掩盖真正的原因。
    if n_written == 0:
        raise RuntimeError(
            f"没有写出任何结果（{n_failed} 条生成失败）。最后一次错误：{last_err}")
    if n_failed:
        print(f"⚠️  {n_failed} 条生成失败，写出 {n_written} 条。最后一次错误：{last_err}")

    print(f"✅ All data processed and saved successfully. ({n_written} 条)")


def main():
    # 环境变量优先，缺省时回退到占位符，保持原来的手改路径用法。
    model = os.environ.get("PALIGN_MODEL", "your model path")
    json_filename = os.environ.get("PALIGN_INPUT", "input file path")
    file_name = os.environ.get("PALIGN_OUTPUT", "output file path")

    # 原本 max_tokens 也是 32768，等于给输出留了 0 空间：只要 prompt 稍长，
    # prompt+output 必然超过 max_model_len。这里把输出预算和过滤阈值绑在一起。
    max_model_len = int(os.environ.get("PALIGN_MAX_MODEL_LEN", 32768))
    max_new_tokens = int(os.environ.get("PALIGN_MAX_TOKENS", 8192))
    batch_size = int(os.environ.get("PALIGN_BATCH_SIZE", 500))

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    sampling_params = SamplingParams(n=1, temperature=0.6, top_p=0.9,
                                     repetition_penalty=1.05, max_tokens=max_new_tokens)
    llm = LLM(model=model, gpu_memory_utilization=0.8, max_model_len=max_model_len,
              trust_remote_code=True, tensor_parallel_size=1)
    process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params,
                 max_model_len=max_model_len, reserve_output=max_new_tokens)

if __name__ == "__main__":
    main()
