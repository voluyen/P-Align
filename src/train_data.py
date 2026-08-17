"""Dataset + collator for P-ALIGN SFT.

Reads the alpaca-format D_align written by build_align_dataset.py:

    {"instruction": "Please reason step by step, ...",
     "input":       "<the question>",
     "output":      "<Begin_of_Prefix>...<End_of_Prefix><continuation>"}

The user turn is `instruction + "\n" + input`, which is how LLaMA-Factory
joins the two alpaca fields -- worth keeping identical, since the authors
trained through LLaMA-Factory.

Loss follows the paper's Eq. 2: negative log-likelihood over the supervision
target only. The prompt is masked with -100 so no gradient flows through it.

    input_ids = [chat-templated prompt] + [output] + [eos]
    labels    = [-100 ... -100]         + [output] + [eos]
"""

import json
import torch
from torch.utils.data import Dataset


class AlignSFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=8192,
                 max_prompt_length=1024, truncate_overlong=False, verbose=True):
        self.tok = tokenizer
        self.max_length = max_length
        self.samples = []

        n_total = n_dropped = n_prompt_trunc = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n_total += 1
                row = json.loads(line)

                # alpaca: LLaMA-Factory nối instruction và input bằng "\n"
                query = row["instruction"]
                if row.get("input", "").strip():
                    query = f"{query}\n{row['input']}"

                prompt_ids = self._encode_prompt(query)
                if len(prompt_ids) > max_prompt_length:
                    prompt_ids = prompt_ids[:max_prompt_length]
                    n_prompt_trunc += 1

                target_ids = self.tok(row["output"], add_special_tokens=False)["input_ids"]
                if self.tok.eos_token_id is not None:
                    target_ids = target_ids + [self.tok.eos_token_id]

                total = len(prompt_ids) + len(target_ids)
                if total > max_length:
                    if not truncate_overlong:
                        # Dropping beats truncating: a clipped target loses the final
                        # \boxed{} answer and the EOS, which trains the model never to stop.
                        n_dropped += 1
                        continue
                    target_ids = target_ids[: max_length - len(prompt_ids)]

                input_ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + target_ids
                self.samples.append((input_ids, labels))

        if verbose:
            lens = [len(x) for x, _ in self.samples]
            print(f"[data] {path}")
            print(f"[data] loaded {len(self.samples)}/{n_total} samples "
                  f"(dropped {n_dropped} over --max-length {max_length}, "
                  f"{n_prompt_trunc} prompts truncated)")
            if lens:
                lens_sorted = sorted(lens)
                print(f"[data] token length  mean={sum(lens)/len(lens):.0f}  "
                      f"p50={lens_sorted[len(lens)//2]}  "
                      f"p95={lens_sorted[int(len(lens)*0.95)]}  max={lens_sorted[-1]}")
            if n_dropped:
                print(f"[data] ⚠️  {n_dropped} samples dropped. Raise --max-length or pass "
                      f"--truncate-overlong if that loss is unacceptable.")

    def _encode_prompt(self, prompt: str):
        msgs = [{"role": "user", "content": prompt}]
        try:
            # enable_thinking=False mirrors the rest of the repo: reasoning is supplied
            # as a prefix, the model must not open its own thinking block.
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class Collator:
    """Right-pads a batch to its own longest sequence."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x) for x, _ in batch)
        input_ids, labels, attn = [], [], []
        for ids, lab in batch:
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
