"""
Two-step data pipeline for synthesizing agent training data, following RULER's design.

Step 1 (synthesize):  generate a polluted long text with hidden "useful information",
                      together with a query (BaseTask.generate_input_output).
Step 2 (annotate):    from the raw material, extract the useful information located by
                      the query, then derive the final answer from it (extraction + outputs
                      are produced in the same pass; both are saved as gold fields).

Usage:
    python -m datamaker.prepare \
        --save_dir ./output \
        --config configs/niah.yaml \
        --task niah_single_1 \
        --max_seq_length 4096 \
        --num_samples 10 \
        --tokenizer_path gpt2 \
        --tokenizer_type hf
"""
import argparse
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamaker.constants import TASKS, TEMPLATES
from datamaker.manifest import write_manifest
from datamaker.tokenizer import select_tokenizer
from datamaker.tasks import build_task


def main():
    parser = argparse.ArgumentParser(description='Synthesize RULER-style two-step training data.')
    parser.add_argument('--save_dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='task complexity yaml')
    parser.add_argument('--task', type=str, required=True, help='task name in the yaml')
    parser.add_argument('--subset', type=str, default='train', help='validation or train')
    parser.add_argument('--tokenizer_path', type=str, default='', help='path/name of the tokenizer ('' = whitespace split)')
    parser.add_argument('--tokenizer_type', type=str, default='none', help='[Options] hf, nemo, none')
    parser.add_argument('--max_seq_length', type=int, required=True, help='input tokens + tokens_to_generate')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--model_template_type', type=str, default='base', help='Options in datamaker/constants.py')
    parser.add_argument('--remove_newline_tab', action='store_true')
    parser.add_argument('--data_dir', type=str, default='', help='dir with squad.json/hotpotqa.json for qa task')
    args = parser.parse_args()

    import random
    import numpy as np
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    with open(args.config, 'r') as f:
        tasks_customized = yaml.safe_load(f)
    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in {args.config}')

    config = tasks_customized[args.task]
    task_base = TASKS[config['task']]
    full_config = {**task_base, **config.get('args', {})}

    tokenizer = select_tokenizer(args.tokenizer_type, args.tokenizer_path)
    task = build_task(config['task'], full_config, tokenizer, args, task_name=args.task)

    # QA-style tasks can be skipped when their dataset file is missing;
    # produce an empty file so downstream (split/train) also skips gracefully.
    if getattr(task, 'skipped', False):
        print(f'[warn] {args.task}: required dataset file missing, skipping generation '
              f'(wrote 0 samples to {args.save_dir}/{args.task}/{args.subset}.jsonl)')
        save_file = task.save(args.save_dir, args.subset, [])
        write_manifest(save_file, [])
        print(f'Wrote 0 samples to {save_file}')
        return

    # prepend model chat template to task template (answer_prefix appended so the
    # final answer is seeded, later split off into its own field)
    model_template = TEMPLATES.get(args.model_template_type, '{task_template}')
    task.template = model_template.format(task_template=task.template) + task.answer_prefix

    start_time = time.time()
    samples = task.generate_samples(num_samples=args.num_samples, max_seq_length=args.max_seq_length)

    save_file = task.save(args.save_dir, args.subset, samples)
    write_manifest(save_file, samples)
    print(f'Wrote {len(samples)} samples to {save_file}')
    print(f'Used time: {round((time.time() - start_time) / 60, 1)} minutes')


if __name__ == '__main__':
    main()