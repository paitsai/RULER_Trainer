#!/usr/bin/env bash
# One-shot pipeline: generate synthetic data for every task in tasks.yaml, then
# train both agents per task. Tweak the CONFIG block below.
#
# Usage:
#   bash start.sh                    # full run with defaults
#   bash start.sh --num_samples 200  # override a single knob (forwarded to prepare.py)
#
# Per-task notes (from AGENTS.md):
#   - CWE has a long extraction (~500 answer tokens) and needs short-context data
#     so that prompt + answer fits under --max_length. We pass --max_seq_length
#     1024 for CWE specifically; everything else uses 4096.
#   - --tokenizer_path gpt2 --tokenizer_type hf is required for accurate length
#     control (the default whitespace tokenizer over-counts).

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SAVE_DIR="datamaker/output"
CONFIG="datamaker/configs/tasks.yaml"
OUTPUT_DIR="checkpoints/ruler_dual_agent"

TOKENIZER_PATH="${TOKENIZER_PATH:-gpt2}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-hf}"

NUM_SAMPLES="${NUM_SAMPLES:-4096}"         # total samples per task (split 3/4 train + 1/4 test)
TRAIN_FRACTION="${TRAIN_FRACTION:-0.75}"    # first 3/4 -> train.jsonl; remaining 1/4 -> test.jsonl
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"    # datamaker --max_seq_length (default); override per-task via MAX_SEQ_LENGTH_OVERRIDES
MAX_SEQ_LENGTH_OVERRIDES="${MAX_SEQ_LENGTH_OVERRIDES:-cwe:1024,cwe_hard:1024}"  # comma-separated "task:len" pairs (e.g. "niah_multikey_1:8192")
MAX_LENGTH="${MAX_LENGTH:-2304}"            # train --max_length; >= prompt + answer
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-}"  # empty = use full training set
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-5e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
GPUS="${GPUS:-1}"                            # torchrun --nproc_per_node
GRAD_CKPT_FLAG="${GRAD_CKPT_FLAG:-}"         # set to "--gradient_checkpointing" for 1.7B+

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -f "$CONFIG" ] || fail "missing config: $CONFIG"
command -v torchrun >/dev/null 2>&1 || fail "torchrun not found on PATH (activate the 'mllm' env?)"

# Build the list of task names from the yaml (one yaml key per task).
TASKS=$(awk '
    /^[a-zA-Z_][a-zA-Z0-9_]*:$/ && !/args:/ { gsub(":$",""); print }
' "$CONFIG")
[ -n "$TASKS" ] || fail "no task names parsed from $CONFIG"

# Per-task max_seq_length override (see CWE note above).
# Format: "task:len,task:len,..." — unset to use MAX_SEQ_LENGTH for everything.
declare -A MAX_SEQ_OVERRIDES=()
if [ -n "$MAX_SEQ_LENGTH_OVERRIDES" ]; then
    IFS=',' read -ra _pairs <<< "$MAX_SEQ_LENGTH_OVERRIDES"
    for pair in "${_pairs[@]}"; do
        t="${pair%%:*}"
        l="${pair#*:}"
        MAX_SEQ_OVERRIDES["$t"]="$l"
    done
fi
max_seq_for() {
    if [ -n "${MAX_SEQ_OVERRIDES[$1]+x}" ]; then
        echo "${MAX_SEQ_OVERRIDES[$1]}"
    else
        echo "$MAX_SEQ_LENGTH"
    fi
}

# ---------------------------------------------------------------------------
# 1) Generate data for every task
# ---------------------------------------------------------------------------
log "Stage 1: generate data  (save_dir=$SAVE_DIR, num_samples=$NUM_SAMPLES, train=$TRAIN_FRACTION)"
mkdir -p "$SAVE_DIR"

# Always start from a clean slate so the 3/4 / 1/4 split is deterministic.
rm -rf "$SAVE_DIR"

for task in $TASKS; do
    max_seq="$(max_seq_for "$task")"
    log "[data] $task  (max_seq_length=$max_seq)"
    # Generate one full pool; we split into train.jsonl / test.jsonl below
    # to avoid re-running the (RNG-consuming) synthesis for the held-out set.
    python -m datamaker.prepare \
        --save_dir "$SAVE_DIR" \
        --config   "$CONFIG" \
        --task     "$task" \
        --max_seq_length "$max_seq" \
        --num_samples    "$NUM_SAMPLES" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --tokenizer_type "$TOKENIZER_TYPE" \
        --data_dir datamaker/data \
        "$@"

    # Split the freshly-written train.jsonl into train (first 3/4) and test (last 1/4).
    task_dir="$SAVE_DIR/$task"
    pool="$task_dir/train.jsonl"
    [ -f "$pool" ] || fail "expected $pool after prepare"
    n_total=$(wc -l < "$pool")
    # floor(n_total * TRAIN_FRACTION)
    n_train=$(awk -v t="$n_total" -v f="$TRAIN_FRACTION" 'BEGIN{print int(t*f)}')
    n_test=$(( n_total - n_train ))
    log "[split] $task  total=$n_total  train=$n_train  test=$n_test"

    # Move the held-out tail to test.jsonl; rewrite train.jsonl with the head only.
    tail -n "$n_test" "$pool" > "$task_dir/test.jsonl"
    head -n "$n_train" "$pool" > "$task_dir/train.jsonl.tmp"
    mv "$task_dir/train.jsonl.tmp" "$task_dir/train.jsonl"

    # Sanity check: any line in test.jsonl must NOT appear in train.jsonl
    overlap=$(comm -12 <(sort "$task_dir/train.jsonl") <(sort "$task_dir/test.jsonl") | wc -l)
    [ "$overlap" -eq 0 ] || fail "train/test overlap for $task: $overlap lines"
done

# ---------------------------------------------------------------------------
# 2) Train each task (both agents)
# ---------------------------------------------------------------------------
log "Stage 2: train dual-agent LoRAs  (output_dir=$OUTPUT_DIR, max_length=$MAX_LENGTH)"
mkdir -p "$OUTPUT_DIR"

# Comma-separated list for the train script.
TASK_CSV=$(echo "$TASKS" | paste -sd, -)

torchrun --nproc_per_node="$GPUS" train/lora_train_ruler.py \
    --data_root              "$SAVE_DIR" \
    --tasks                  "$TASK_CSV" \
    --agents                 extract,answer \
    --model_name             "$MODEL_NAME" \
    --max_length             "$MAX_LENGTH" \
    --batch_size             "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lr                     "$LR" \
    --num_epochs             "$NUM_EPOCHS" \
    --lora_r                 "$LORA_R" \
    --lora_alpha             "$LORA_ALPHA" \
    --val_jsonl              test.jsonl \
    --max_samples_per_task   "${MAX_SAMPLES_PER_TASK:-2147483647}" \
    --output_dir             "$OUTPUT_DIR" \
    $GRAD_CKPT_FLAG

log "Done."
echo "  data:   $SAVE_DIR/<task>/train.jsonl"
echo "  ckpts:  $OUTPUT_DIR/<task>/agent_{extract,answer}/{final,best,step_N}"