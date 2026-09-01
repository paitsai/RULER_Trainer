# AGENTS.md

Two-stage pipeline: `datamaker/` synthesizes RULER-style training data, `train/` fine-tunes a dual-agent LoRA system on it. No test suite — verify by running the commands below on a small sample.

## Key commands

```bash
# 1. Generate synthetic data (one task config)
python -m datamaker.prepare \
    --save_dir datamaker/output \
    --config datamaker/configs/tasks.yaml \
    --task niah_multikey_1 \
    --max_seq_length 4096 --num_samples 100 \
    --tokenizer_path gpt2 --tokenizer_type hf

# 2. Train both agents (extract + answer) per task
torchrun --nproc_per_node=1 train/lora_train_ruler.py \
    --data_root datamaker/output \
    --tasks niah_multikey_1,cwe \
    --agents extract,answer \
    --max_length 2304 --max_samples_per_task 32 --output_dir /tmp/pilot
```

## Pipeline structure

- **datamaker**: `BaseTask.generate_input_output()` (synthesize context+extraction+answers) → `estimate_optimal_source()` (binary search on token count vs `max_seq_length`) → JSONL per task. Task registry: `datamaker/tasks/__init__.py`; configs: `datamaker/configs/tasks.yaml`.
- **train**: two agents share one base model with separate LoRAs. agent-1 `extract`: `input → extraction`; agent-2 `answer`: `input + extraction → outputs` (cascade/shared context). LoRA dirs: `{output_dir}/{task}/agent_extract|agent_answer/{final,step_N,best}`.
- Qwen3-Base is used as base model (worked: 0.6B; 1.7B OOMs without `--gradient_checkpointing`).

## Gotchas (verified, easy to trip on)

- **loss=NaN ⇒ max_length too short**: all assistant tokens get truncated, labels end up fully `-100`-masked. `max_length` must cover prompt + answer. With Qwen3 tokenizer, 2K-context data needs `--max_length 2304+`; CWE has a long extraction (~500 answer tokens) and needs short-context data (`max_seq_length 1024`).
- **Data files are gitignored** (`*.jsonl` in `.gitignore`): generated data and checkpoints won't appear in `git status`.
- **Train prefers `train_short.jsonl` over `train.jsonl`** per task dir (`iter_train_files`). If you regenerate data with `--save_dir datamaker/output`, it writes `train.jsonl`; the short-context smoke files are manual renames.
- **datamaker `input` ≠ full prompt**: the answer prefix is split off into the `answer_prefix` field during generation; train/data.py rebuilds the final assistant response from `answer_prefix + outputs`.
- **Tokenization**: default `--tokenizer_type none` counts whitespace — use `--tokenizer_path gpt2 --tokenizer_type hf` for accurate length control.
- **Qwen3-Base chat template hides "thinking" blocks**: `preprocess()` in train/lora_train_ruler.py must render prompt with `add_generation_prompt=True`, append assistant content + `<|im_end|>` manually, and mask via prefix-token-length — not by searching for an assistant marker in templated text.
- **QA task** needs real data at `datamaker/data/squad.json` / `hotpotqa.json` (`qa.py`). SQuAD download works; HotpotQA mirrors are flaky/network-limited.
- **Precision**: model loaded in `torch.bfloat16` (not fp16). `--gradient_checkpointing` is opt-in (off by default).

## Conventions

- python 3.10 env `mllm` (`/home/users/xzr/miniconda3/envs/mllm/bin/python`); torch 2.5.1+cu124, 4× RTX 3090.
- Add a new task: subclass `BaseTask`, implement `generate_input_output(num_source, index)` returning `(context, extraction_lines, answers, meta)`, register in `datamaker/tasks/__init__.py`, add config in `datamaker/configs/tasks.yaml`.
