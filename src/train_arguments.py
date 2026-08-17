"""Command-line arguments for P-ALIGN SFT training (DistiLLM-style flags)."""

import argparse
import deepspeed


def get_args():
    p = argparse.ArgumentParser()

    # --- model / data ---
    p.add_argument("--model-path", type=str, required=True,
                   help="HF path of the student model (Qwen2.5-7B-Instruct / Qwen3-8B)")
    p.add_argument("--data-path", type=str, required=True,
                   help="D_align jsonl produced by src/build_align_dataset.py")
    p.add_argument("--save", type=str, required=True, help="checkpoint output dir")
    p.add_argument("--num-workers", type=int, default=2)

    # --- sequence length ---
    # Paper Table 4: P-ALIGN sequences average 5,453 tok (Qwen2.5-7B) / 5,732 (Qwen3-8B).
    p.add_argument("--max-length", type=int, default=8192,
                   help="max total length of prompt + target")
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--truncate-overlong", action="store_true",
                   help="truncate samples longer than --max-length instead of dropping "
                        "them. Off by default: truncation cuts the trailing \\boxed{} "
                        "answer and the EOS token, which teaches the model never to stop.")

    # --- optimisation (paper Sec. 4: lr 5e-5, 3 epochs, LoRA) ---
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1, help="micro-batch per GPU")
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="required at 8k context on a single 40GB card")
    p.add_argument("--attn-impl", type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="flash_attention_2 saves the most activation memory at 8k, "
                        "but needs the flash-attn package")

    # --- LoRA ---
    p.add_argument("--use-lora", action="store_true",
                   help="LoRA fine-tuning (what the paper does). Without this flag the "
                        "run is a full fine-tune, which needs ZeRO-3 + CPU offload.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # --- runtime ---
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=-1,
                   help="save every N optimizer steps; -1 = only at each epoch end")
    p.add_argument("--local_rank", type=int, default=-1)

    # adds --deepspeed / --deepspeed_config
    p = deepspeed.add_config_arguments(p)
    return p.parse_args()
