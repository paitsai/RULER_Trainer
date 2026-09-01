"""
Data utilities for the dual-agent RULER training pipeline.

Two agents share the same base model but have different LoRAs:
  agent-1 (extract):  input                       -> extraction
  agent-2 (answer):   input + "\n" + extraction   -> outputs

The second agent's prompt is the first agent's prompt (with its assistant
response) plus an additional instruction that asks for the final answer,
so the two share an evolving "context window".
"""
import json
import random
from pathlib import Path
from typing import Iterator, List, Optional

# ensure datamaker is importable when this file is used standalone
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from torch.utils.data import Dataset


SYSTEM_PROMPT_EXTRACT = (
    "You are a careful information-extraction agent. Your job is to locate and "
    "quote the relevant pieces of information in the provided text that the next "
    "agent will need in order to answer the question. Be precise; include every "
    "piece that could be useful, and nothing extra."
)

SYSTEM_PROMPT_ANSWER = (
    "You are a precise answering agent. The first agent has already extracted the "
    "useful information from the long text. Combine the original context and the "
    "extracted information to produce the final answer. Be concise and follow the "
    "expected output format."
)

# instruction appended after the extraction so the answer agent knows what to do.
ANSWER_INSTRUCTION = (
    "\n\nNow, using the original text and the extracted information above, "
    "produce the final answer. Be concise."
)


def load_ruler_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages_for_extract(example: dict) -> list:
    """agent-1: input -> extraction."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_EXTRACT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["extraction"]},
    ]


def build_messages_for_answer(example: dict) -> list:
    """agent-2: input + extraction -> outputs.

    The user message contains both the original input and the agent-1 extraction,
    modelling the shared context window across agents.
    """
    augmented_user = (
        example["input"]
        + "\n\n--- Extracted information from agent 1 ---\n"
        + (example["extraction"] or "(no information extracted)")
        + ANSWER_INSTRUCTION
    )
    final = example["answer_prefix"].strip() + " " + ", ".join(example["outputs"])
    return [
        {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
        {"role": "user", "content": augmented_user},
        {"role": "assistant", "content": final},
    ]


class RulerDualAgentDataset(Dataset):
    """Yields per-agent training samples from a RULER-style jsonl.

    Set ``agent`` to either 'extract' or 'answer' to pick the training objective.
    """

    def __init__(self, examples: List[dict], agent: str, tokenizer, max_length: int,
                 model_name: str):
        assert agent in ("extract", "answer")
        self.examples = examples
        self.agent = agent
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.agent == "extract":
            messages = build_messages_for_extract(ex)
        else:
            messages = build_messages_for_answer(ex)
        return {
            "messages": messages,
            "task": ex.get("task", "unknown"),
            "index": ex.get("index", idx),
        }


def iter_train_files(data_root: Path, task_names: Optional[List[str]] = None) -> Iterator[Path]:
    """Yield jsonl files under <data_root>/<task>/{train_short,train}.jsonl.

    Prefers train_short.jsonl (short-context smoke-test data) when present;
    falls back to train.jsonl. If task_names is provided, only those tasks are considered.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f'data root not found: {data_root}')
    task_root = data_root
    for task_dir in sorted(task_root.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_names is not None and task_dir.name not in task_names:
            continue
        for name in ("train_short.jsonl", "train.jsonl"):
            f = task_dir / name
            if f.exists():
                yield f
                break