# AGENTS.md

Two-stage pipeline: `datamaker/` synthesizes RULER-style training data, `train/` fine-tunes a dual-agent LoRA system on it. No test suite — verify by running commands on a small sample (`NUM_SAMPLES=60 bash start.sh`).

## Key commands

```bash
# One-shot pipeline: generate data for all tasks + train. This is the primary flow.
bash start.sh                        # 4096 samples/task -> train.jsonl (3/4) + test.jsonl (1/4)
NUM_SAMPLES=200 bash start.sh        # smoke run (start.sh wipes datamaker/output first)

# Focused manual steps (one task config, as used before start.sh existed)
python -m datamaker.prepare \
    --save_dir datamaker/output \
    --config datamaker/configs/tasks.yaml \
    --task niah_multikey_1 \
    --max_seq_length 4096 --num_samples 100 \
    --tokenizer_path gpt2 --tokenizer_type hf

torchrun --nproc_per_node=1 train/lora_train_ruler.py \
    --data_root datamaker/output \
    --tasks niah_multikey_1,cwe \
    --agents extract,answer \
    --max_length 2304 --max_samples_per_task 32 \
    --val_jsonl test.jsonl --output_dir /tmp/pilot
```

## Pipeline structure

- **datamaker**: `BaseTask.generate_input_output()` (synthesize context+extraction+answers) → `estimate_optimal_source()` (binary search on token count vs `max_seq_length`) → JSONL per task. Task registry: `datamaker/tasks/__init__.py`; configs: `datamaker/configs/tasks.yaml`.
- **train**: two agents share one base model with separate LoRAs. agent-1 `extract`: `input → extraction`; agent-2 `answer`: `input + extraction → outputs` (cascade/shared context). LoRA dirs: `{output_dir}/{task}/agent_extract|agent_answer/{final,step_N,best}`.
- **Data split (start.sh)**: one pool of `NUM_SAMPLES` is generated, then split in place — first `TRAIN_FRACTION` (default 0.75) → `train.jsonl`, tail → `test.jsonl`. Training loads `train.jsonl` fully and evaluates on `test.jsonl` via `--val_jsonl` (no internal `val_ratio` split when it's set).
- Qwen3-Base is used as base model (worked: 0.6B; 1.7B OOMs without `--gradient_checkpointing`).

## start.sh knobs (env vars)

- `NUM_SAMPLES` (4096), `TRAIN_FRACTION` (0.75), `MAX_SAMPLES_PER_TASK` (empty = full train set).
- `MAX_SEQ_LENGTH` (4096) = datamaker generation length; per-task override via `MAX_SEQ_LENGTH_OVERRIDES="cwe:1024,cwe_hard:1024"` (`"task:len"` comma-separated).
- `MAX_LENGTH` (2304) = train seq length, must >= prompt + answer (see loss=NaN gotcha).
- `GPUS` (1), `GRAD_CKPT_FLAG` (set to `--gradient_checkpointing` for 1.7B), `MODEL_NAME`, `LR`, `NUM_EPOCHS`, `LORA_R`/`LORA_ALPHA`.
- Extra args after the script name forward to `prepare.py` (last-wins): `bash start.sh --random_seed 7`.

## Gotchas (verified, easy to trip on)

- **start.sh `rm -rf datamaker/output` at stage 1** — anything hand-placed there (e.g. renamed smoke files) is wiped. Keep manual data in a different `--save_dir`.
- **loss=NaN ⇒ `max_length` too short**: all assistant tokens get truncated, labels end up fully `-100`-masked. `MAX_LENGTH` must cover prompt + answer. With Qwen3 tokenizer, 2K-context data needs 2304+; CWE has a long extraction (~500 answer tokens) and needs short-context generation (`MAX_SEQ_LENGTH_OVERRIDES` cwe→1024).
- **Data files are gitignored** (`*.jsonl` in `.gitignore`): generated data and checkpoints won't appear in `git status`.
- **Manual runs: `train.jsonl` vs `train_short.jsonl`** (`iter_train_files` prefers `train_short.jsonl`). Only matters outside start.sh, since start.sh regenerates and splits fresh.
- **datamaker `input` ≠ full prompt**: the answer prefix is split off into the `answer_prefix` field during generation; train/data.py rebuilds the final assistant response from `answer_prefix + outputs`.
- **Tokenization**: default `--tokenizer_type none` counts whitespace — `start.sh` and focused runs use `--tokenizer_path gpt2 --tokenizer_type hf` for accurate length control.
- **Qwen3-Base chat template hides "thinking" blocks**: `preprocess()` in train/lora_train_ruler.py must render prompt with `add_generation_prompt=True`, append assistant content + `<|im_end|>` manually, and mask via prefix-token-length — not by searching for an assistant marker in templated text.
- **QA task** needs real data at `datamaker/data/squad.json` / `hotpotqa.json` (`qa.py`). SQuAD download works; HotpotQA mirrors are flaky/network-limited. If the file is missing, `qa.py` warns and sets `task.skipped=True`; `prepare.py` then writes an empty `train.jsonl` and the rest of the pipeline (split, train) skips the task silently.
- **Precision**: model loaded in `torch.bfloat16` (not fp16). `--gradient_checkpointing` is opt-in (off by default).
- **`--val_jsonl` fallback**: if the eval file (default `test.jsonl`) is missing for a task, training silently falls back to the internal `val_ratio` split and prints a warning.

## Conventions

- python 3.10 env `mllm` (`/home/users/xzr/miniconda3/envs/mllm/bin/python`); torch 2.5.1+cu124, 4× RTX 3090.
- Add a new task: subclass `BaseTask`, implement `generate_input_output(num_source, index)` returning `(context, extraction_lines, answers, meta)`, register in `datamaker/tasks/__init__.py`, add config in `datamaker/configs/tasks.yaml`.