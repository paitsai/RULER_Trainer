"""
Multi-GPU dual-agent LoRA training for synthetic RULER tasks.

For each RULER task, two agents are trained on the same base model with separate
LoRA adapters:

    agent-1 (extract):  input                 -> extraction   (collect useful info)
    agent-2 (answer):   input + extraction    -> outputs      (final answer, shares context)

The answer agent's prompt is the extract agent's prompt plus the extracted
information and its own instruction, i.e. the second agent benefits from an
evolving shared context window.

Launch via torchrun:
    torchrun --nproc_per_node=4 train/lora_train_ruler.py \
        --data_root datamaker/output \
        --tasks niah_multikey_1 vt_1 cwe fwe \
        --output_dir checkpoints/ruler_dual_agent
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

# allow `python train/lora_train_ruler.py` and torchrun from anywhere
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from train.data import (
    RulerDualAgentDataset,
    iter_train_files,
    load_ruler_jsonl,
    build_messages_for_extract,
    build_messages_for_answer,
)

AGENT_EXTRACT = "extract"
AGENT_ANSWER = "answer"
AGENTS = [AGENT_EXTRACT, AGENT_ANSWER]
AGENT_SUBDIR = {AGENT_EXTRACT: "agent_extract", AGENT_ANSWER: "agent_answer"}


def parse_args():
    p = argparse.ArgumentParser(description="Dual-agent LoRA fine-tuning on synthetic RULER tasks")
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--data_root", type=str, default="datamaker/output",
                   help="Root dir containing <task>/train.jsonl produced by datamaker")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task names; default: all tasks found under data_root")
    p.add_argument("--agents", type=str, default="extract,answer",
                   help="Comma-separated agents to train ('extract' and/or 'answer')")
    p.add_argument("--max_length", type=int, default=4608,
                   help="Max sequence length (RULER inputs up to 4096 + extraction)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.08)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce activation memory (off by default)")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Fraction of each task's samples held out for evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples_per_task", type=int, default=None,
                   help="Limit training samples per task (useful for smoke tests)")
    p.add_argument("--val_jsonl", type=str, default=None,
                   help="Explicit held-out eval file (e.g. <task>/test.jsonl). When set, "
                        "no samples are carved out of the training file for validation.")
    p.add_argument("--output_dir", default=None,
                   help="Output root; per-task per-agent LoRA dirs are created inside")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers (same as GSM8K template)
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main_process(rank):
    return rank == 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Tokenization (shared by both agents)
# ---------------------------------------------------------------------------

def preprocess(messages_list, tokenizer, max_length, model_name):
    """Build training samples with masked user/system tokens.

    For each conversation we render (system+user) with `add_generation_prompt=True`
    to get a deterministic prompt prefix, then append the assistant content +
    the model's eos token by hand. This avoids base-model chat templates that
    inject hidden blocks (e.g. Qwen3's <think>…</think>) and ensures the
    assistant tokens are properly included.
    """
    if 'qwen' in model_name.lower():
        assistant_marker = "<|im_start|>assistant\n"
        assistant_end = "<|im_end|>"
    elif 'llama' in model_name.lower():
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        assistant_end = "<|eot_id|>"
    else:
        raise ValueError(f"Unknown model type for chat template: {model_name}")

    texts, prompt_lens = [], []
    for messages in messages_list:
        # split off the assistant message; everything else is the prompt
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        assistant_msg = next(m for m in messages if m["role"] == "assistant")
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + assistant_msg["content"] + assistant_end
        texts.append(full_text)
        prompt_lens.append(len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]))

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = tokenized["input_ids"]
    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


def make_dataloaders(examples, tokenizer, args, rank, world_size, agent, val_examples=None):
    random.Random(args.seed).shuffle(examples)
    if val_examples is not None:
        train_examples, val_examples = examples, val_examples
    else:
        n_val = int(len(examples) * args.val_ratio)
        train_examples = examples[n_val:]
        val_examples = examples[:n_val]

    train_ds = RulerDualAgentDataset(train_examples, agent, tokenizer, args.max_length, args.model_name)
    val_ds = RulerDualAgentDataset(val_examples, agent, tokenizer, args.max_length, args.model_name)

    def _collate(batch):
        messages = [b["messages"] for b in batch]
        encoded = preprocess(messages, tokenizer, args.max_length, args.model_name)
        return encoded

    sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=(sampler is None), drop_last=False,
                              collate_fn=_collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=_collate, num_workers=2)
    return train_loader, val_loader, len(train_examples), len(val_examples)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def normalize(text):
    return re.sub(r"[^a-zA-Z0-9]", "", text.lower())


def evaluate_answer_accuracy(model, tokenizer, val_examples, device, max_length, max_eval_samples=50):
    """Feed a few val examples through the full two-agent chain (greedy) and
    check whether the answer agent's generation contains the gold outputs."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        sampled = random.Random(0).sample(val_examples, min(max_eval_samples, len(val_examples)))
        for ex in sampled:
            # --- agent-1: extract (prompt only; no gold answer leaked) ---
            m1 = [m for m in build_messages_for_extract(ex) if m["role"] != "assistant"]
            text1 = tokenizer.apply_chat_template(m1, tokenize=False, add_generation_prompt=True)
            inputs1 = tokenizer(text1, return_tensors="pt", truncation=True,
                                max_length=max_length).to(device)
            out1 = model.generate(**inputs1, max_new_tokens=256, do_sample=False)
            extraction_pred = tokenizer.decode(out1[0][inputs1["input_ids"].shape[1]:], skip_special_tokens=True)

            # --- agent-2: answer (prompt only with predicted extraction) ---
            aug = (ex["input"] + "\n\n--- Extracted information from agent 1 ---\n"
                   + (extraction_pred or "(nothing)") + "\n\nNow answer the question concisely.")
            m2 = [
                {"role": "system", "content": "You are a precise answering agent."},
                {"role": "user", "content": aug},
            ]
            text2 = tokenizer.apply_chat_template(m2, tokenize=False, add_generation_prompt=True)
            inputs2 = tokenizer(text2, return_tensors="pt", truncation=True,
                                max_length=max_length).to(device)
            out2 = model.generate(**inputs2, max_new_tokens=128, do_sample=False)
            pred = tokenizer.decode(out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True)

            golds = [normalize(a) for a in ex["outputs"]]
            pred_norm = normalize(pred)
            if all(g in pred_norm for g in golds):
                correct += 1
            total += 1
    model.train()
    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Model loading / LoRA
# ---------------------------------------------------------------------------

def load_lora_model(model_name, device, args):
    # bf16 has the same exponent range as fp32 -> avoids loss=NaN on 1.7B+ models
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=None,
    ).to(device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.train()
    return model


# ---------------------------------------------------------------------------
# Training loop (one agent)
# ---------------------------------------------------------------------------

def train_agent(args, rank, world_size, local_rank, task_name, agent, examples, val_examples=None):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        print(f"\n{'='*60}\nTraining agent-{agent} on task '{task_name}' "
              f"({len(examples)} samples)\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_lora_model(args.model_name, device, args)
    if is_main_process(rank):
        model.print_trainable_parameters()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    unwrapped = model.module if world_size > 1 else model

    train_loader, val_loader, n_train, n_val = make_dataloaders(
        examples, tokenizer, args, rank, world_size, agent, val_examples=val_examples)

    steps_per_epoch = max(len(train_loader) // args.gradient_accumulation_steps, 1)
    total_steps = steps_per_epoch * args.num_epochs

    if is_main_process(rank):
        print(f"  train={n_train} val={n_val} batches/epoch={len(train_loader)} "
              f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    output_dir = Path(args.output_dir) / task_name / AGENT_SUBDIR[agent]
    output_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(train_loader)
    loss_history = []
    eval_history = []
    best_accuracy = 0.0
    pbar = tqdm(total=total_steps, desc=f"[{task_name}/{agent}]",
                disable=not is_main_process(rank))

    for step in range(1, total_steps + 1):
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_history.append(accumulated_loss)
        pbar.set_postfix({"loss": f"{accumulated_loss:.4f}",
                          "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        pbar.update(1)

        if is_main_process(rank) and step % args.eval_interval == 0:
            if agent == AGENT_ANSWER:
                accuracy = evaluate_answer_accuracy(unwrapped, tokenizer, val_loader.dataset.examples,
                                                    device, args.max_length, max_eval_samples=20)
            else:
                accuracy = 0.0  # extraction accuracy is measured downstream via agent-2
            eval_history.append((step, accuracy))
            tqdm.write(f"\n[{task_name}/{agent} step {step}] loss={accumulated_loss:.4f} "
                       f"answer_accuracy={accuracy:.2%}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_path = output_dir / "best"
                best_path.mkdir(parents=True, exist_ok=True)
                unwrapped.save_pretrained(str(best_path))

        if is_main_process(rank) and (step % args.save_interval == 0 or step == total_steps):
            save_path = output_dir / f"step_{step}"
            save_path.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(str(save_path))
            tqdm.write(f"[{task_name}/{agent}] saved LoRA -> {save_path}")

    if is_main_process(rank):
        final_path = output_dir / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        summary = {
            "task": task_name,
            "agent": agent,
            "train_size": n_train,
            "val_size": n_val,
            "total_steps": total_steps,
            "epochs": args.num_epochs,
            "best_accuracy": best_accuracy,
            "final_loss": loss_history[-1] if loss_history else None,
        }
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{task_name}/{agent}] done. LoRA -> {final_path}, best_acc={best_accuracy:.2%}")

    pbar.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    torch.manual_seed(args.seed + rank)

    # resolve tasks
    task_files = list(iter_train_files(args.data_root,
                                       args.tasks.split(",") if args.tasks else None))
    if not task_files:
        raise SystemExit(f"No task jsonl found under {args.data_root}")

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    for a in agents:
        if a not in AGENTS:
            raise ValueError(f"Unknown agent '{a}'; choose from {AGENTS}")

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"ruler_dual_agent_{ts}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if is_main_process(rank):
        print(f"GPUs={world_size} tasks={[f.name for f in task_files]} agents={agents}")
        print(f"data_root={args.data_root} output_dir={args.output_dir} max_length={args.max_length}")

    for task_file in task_files:
        task_name = task_file.parent.name
        examples = load_ruler_jsonl(task_file)
        if args.max_samples_per_task is not None:
            examples = examples[: args.max_samples_per_task]
        if len(examples) < 2:
            if is_main_process(rank):
                print(f"skip {task_name}: too few samples ({len(examples)})")
            continue
        # explicit held-out set (e.g. test.jsonl) for eval when requested
        val_examples = None
        if args.val_jsonl:
            val_file = Path(args.val_jsonl).resolve()
            if task_file.parent != val_file.parent:
                val_file = task_file.parent / args.val_jsonl
            if val_file.exists():
                val_examples = load_ruler_jsonl(val_file)
            elif is_main_process(rank):
                print(f"[warn] val file not found for {task_name}: {val_file} (fall back to internal split)")
        for agent in agents:
            train_agent(args, rank, world_size, local_rank, task_name, agent, examples,
                        val_examples=val_examples)

    cleanup_distributed()


if __name__ == "__main__":
    main()