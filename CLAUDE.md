# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper *Long-Chain Reasoning Distillation via Adaptive Prefix Alignment* (P-ALIGN). It is a small collection of standalone scripts — not a package, not a library. There is no test suite, no linter config, no CI, and no `__init__.py`. Every script is run directly with `python src/<file>.py` from the repo root.

Model/dataset artifacts live on HuggingFace (`qizheyanger/P-ALIGN`); raw datasets are not redistributed (see `data/raw/README.md`).

## Environment

Two separate conda envs by design:

- `P-ALIGN` (python 3.10) — runs everything in `src/`. Key pins: torch 2.8.0, vllm 0.11.0, transformers 4.57.1, math-verify 0.8.0.
- `llama_factory` — SFT training only, via an external [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) checkout. Training is *not* implemented in this repo.

`requirements.txt` is a **conda export** (`name=version=build` lines, `# platform: linux-64` header), not a pip requirements file. The README's `pip install -r requirements.txt --force-reinstall --no-deps --no-cache-dir` works only because `--no-deps` tolerates the format; prefer `conda create --name <env> --file requirements.txt` on linux-64. Note `oat_math_grader` (imported by `src/evaluation.py`) is **not** in it and must be installed separately.

## Pipeline

Four sequential stages. Each stage reads the previous stage's JSONL output; the wiring between them is hardcoded file paths, not arguments.

1. **Prefix truncation** — `src/binary_search.py` (via `scripts/Prefix_truncation.sh`)
   Splits each teacher Long-CoT into sentences, then binary-searches for the shortest prefix the *student* model judges sufficient. Sufficiency is a self-evaluation prompt whose answer is parsed by string-matching `[ENOUGH]` / `[NOT_ENOUGH]`. Uses HF `AutoModelForCausalLM.generate` (not vLLM). Appends results line-by-line so a crash keeps partial output.
   In: `{question, Long-CoT, answer, id}` → Out: `{question, answer, sufficient_reasoning, sufficient_sentences, total_sentences, prefix_ratio, is_sufficient, evaluator_response}`

2. **Prefix alignment** — `src/prefix-alignment.py` (via `scripts/Prefix_alignment.sh`)
   Feeds each `sufficient_reasoning` back as a "*Prefix*:" draft in the prompt and has the student continue to a `\boxed{}` answer. vLLM, `max_model_len=32768`. Has resume support: re-reads the output file and skips any `question` already present, so it is safe to re-run over a partial file.

3. **Inference** — `src/test.py` (via `scripts/Inference.sh`)
   The only script with a real CLI (`--model`, `--input_files`/`--output_files` as parallel lists, `--n`, `--temperature`, `--max_tokens`, …). Auto-detects the problem field from `problem|question|input|content` and the answer field from `answer|target|solution|ground_truth`, so it runs unchanged across AIME/AMC/MATH500 formats. `--n 3` produces the three samples that pass@3 is computed over. Note `max_model_len` is bound to `--max_tokens`, so raising output length also raises the context budget.
   Out: `{prompt_ori, answer, output: [str × n]}`

4. **Evaluation** — `src/evaluation.py` (via `scripts/evaluation.sh`)
   Dual grader: symbolic `math_verify` OR'd with `oat_math_grader.boxed_reward_fn` (controlled by `use_oat` / `any_true`). `math_verify` is wrapped in a SIGALRM 10s timeout because sympy can hang on adversarial LaTeX — that wrapper is POSIX-only and no-ops on Windows. Reports pass@3 (any of n correct) and acc@3 (mean over all n).
   Adds: `{label: [0/1 × n], passn, output_ans}`

Committed reference outputs for Qwen2.5-7B are in `data/result/*-output.jsonl` (aime24, aime25, amc12) — already evaluated, so they carry the stage-4 fields. Use these to sanity-check stage 4 without a GPU.

## Known breakage — verify before trusting a script

The scripts and sources are out of sync. Do not assume a path in a `.sh` file resolves:

- `scripts/Prefix_truncation.sh` runs `src/binary_select.py`; the file is `src/binary_search.py`.
- `scripts/Prefix_alignment.sh` runs `src/Prefix_alignment.py`; the file is `src/prefix-alignment.py`.
- `scripts/train.sh` runs `src/train.py`, which does not exist (training is delegated to LLaMA-Factory).
- README says `bash scripts/Evaluation.sh`; the file is `scripts/evaluation.sh`.
- Every script writes to `output/log/`, which is not in the repo and is not created — `mkdir -p output/log` first.
- `src/test.py:20` references an undefined `problem` (should be `item[problem_key]`) — stage 3 raises `NameError` as committed.
- `src/evaluation.py:149` prints an undefined `pass_at_1` — stage 4 raises `NameError` after the real metrics are printed and the output file is written.

The module filename `prefix-alignment.py` contains a hyphen and is therefore not importable; it only works as a script.

## Conventions in this codebase

- **All paths are placeholder strings** (`"your model path"`, `"Path to your input jsonl file"`) hardcoded in `main()` or at module scope. Configuring a run means editing the source, except for `src/test.py`. Preserve that style rather than introducing a config system unless asked.
- Data interchange is always JSONL, one record per line, written incrementally with `flush()`+`fsync()` where resumability matters.
- Console output and comments mix English and Chinese. Match whatever is already in the file you're editing.
- Scripts pin GPUs with `export CUDA_VISIBLE_DEVICES=...` in the `.sh` wrapper and run under `nohup ... &`; vLLM is always `tensor_parallel_size=1`, `gpu_memory_utilization=0.8`.
- Chat templates are applied with `enable_thinking=False` everywhere — the point of the method is to supply reasoning as a prefix, not to let the model emit its own thinking block.
