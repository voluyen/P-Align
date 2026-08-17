"""P-ALIGN supervised fine-tuning: PyTorch + DeepSpeed.

Implements the paper's Eq. 2 on the D_align dataset. This is plain SFT — P-ALIGN
carries no KD loss, all of its contribution lives in how D_align is built
(src/binary_search.py -> src/prefix-alignment.py -> src/build_align_dataset.py).

Launch via scripts/train_sft.sh (torchrun + --deepspeed_config).
"""

import json
import math
import os
import random

import numpy as np
import torch
import deepspeed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_arguments import get_args
from train_data import AlignSFTDataset, Collator


def is_main():
    return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_ds_config(args, num_training_steps, hidden_size):
    """Load the JSON config and fill in everything DeepSpeed cannot infer.

    Note: raw deepspeed.initialize() does NOT resolve "auto" values — that is a
    HuggingFace Trainer feature — so every ZeRO-3 bucket size is set explicitly here.
    """
    with open(args.deepspeed_config, "r") as f:
        cfg = json.load(f)

    cfg["train_micro_batch_size_per_gpu"] = args.batch_size
    cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    cfg["gradient_clipping"] = args.clip_grad

    cfg["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": args.lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
        },
    }
    cfg["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0.0,
            "warmup_max_lr": args.lr,
            "warmup_num_steps": max(1, int(num_training_steps * args.warmup_ratio)),
            "total_num_steps": num_training_steps,
        },
    }

    zero = cfg.get("zero_optimization", {})
    if zero.get("stage") == 3:
        zero["reduce_bucket_size"] = hidden_size * hidden_size
        zero["stage3_prefetch_bucket_size"] = int(0.9 * hidden_size * hidden_size)
        zero["stage3_param_persistence_threshold"] = 10 * hidden_size
        cfg["zero_optimization"] = zero

    return cfg


def save_checkpoint(engine, tokenizer, args, tag):
    out = os.path.join(args.save, tag)
    os.makedirs(out, exist_ok=True)

    if args.use_lora:
        # Save only the adapter; merge later with peft's merge_and_unload().
        if is_main():
            engine.module.save_pretrained(out)
            tokenizer.save_pretrained(out)
    else:
        # Consolidates the ZeRO-3 shards into a single fp16/bf16 state dict.
        engine.save_16bit_model(out, "pytorch_model.bin")
        if is_main():
            engine.module.config.save_pretrained(out)
            tokenizer.save_pretrained(out)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    log(f"[save] {out}")


def main():
    args = get_args()
    deepspeed.init_distributed()
    set_seed(args.seed)
    os.makedirs(args.save, exist_ok=True)

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- data ---
    dataset = AlignSFTDataset(
        args.data_path, tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        truncate_overlong=args.truncate_overlong,
        verbose=is_main(),
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check --data-path and --max-length.")

    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) \
        if torch.distributed.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=Collator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    log(f"[train] {len(dataset)} samples | {steps_per_epoch} opt-steps/epoch "
        f"| {total_steps} total")

    # --- model ---
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    )
    model.config.use_cache = False

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        ))
        if is_main():
            model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Without this the checkpointed blocks receive no grad-requiring input
        # when the base weights are frozen (LoRA), and backward silently no-ops.
        model.enable_input_require_grads()

    hidden_size = getattr(model.config, "hidden_size", 4096)
    ds_config = build_ds_config(args, total_steps, hidden_size)
    params = [p for p in model.parameters() if p.requires_grad]

    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=params, config=ds_config
    )

    # --- train ---
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        engine.train()
        running, seen = 0.0, 0

        for micro_step, batch in enumerate(loader):
            batch = {k: v.to(engine.device) for k, v in batch.items()}
            loss = engine(**batch).loss
            engine.backward(loss)
            engine.step()

            running += loss.item()
            seen += 1

            if engine.is_gradient_accumulation_boundary():
                global_step += 1
                if global_step % args.log_interval == 0:
                    log(f"[epoch {epoch+1}/{args.epochs}] step {global_step}/{total_steps} "
                        f"loss {running/max(seen,1):.4f} "
                        f"lr {engine.get_lr()[0]:.2e}")
                    running, seen = 0.0, 0

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    save_checkpoint(engine, tokenizer, args, f"step_{global_step}")

        save_checkpoint(engine, tokenizer, args, f"epoch_{epoch+1}")

    log("✅ training done")


if __name__ == "__main__":
    main()
