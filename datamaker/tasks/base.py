from abc import ABC, abstractmethod
from pathlib import Path
import json
import random

import numpy as np


class BaseTask(ABC):
    name = 'base'

    def __init__(self, config, tokenizer, args, task_name=None):
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.task_name = task_name or self.name
        self.tokens_to_generate = config.get('tokens_to_generate', 0)
        self.template = config.get('template', '')
        self.answer_prefix = config.get('answer_prefix', '')
        self.extraction_template = config.get('extraction_template', '')

    @property
    def incremental(self):
        return 10

    @abstractmethod
    def generate_input_output(self, num_source: int, index: int = 0):
        """Step 1: synthesize raw material.
        Returns (context: str, extraction: list[str], answers: list[str], meta: dict).
        - context: the polluted long text fed to the model
        - extraction: the "useful information" hidden in context (and their locations)
        - answers: final answers derived from extraction
        - meta: extra per-sample metadata (needle positions etc.)
        """
        raise NotImplementedError

    def format_input(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)

    def format_extraction(self, extraction: list, meta: dict) -> str:
        if not extraction:
            return ''
        return self.extraction_template.format(extraction='\n'.join(extraction))

    def estimate_optimal_source(self, max_seq_length: int, incremental_mul: int = 3):
        # use a larger representative sample to estimate tokens per source unit,
        # otherwise small samples (e.g. all-common words in CWE) overestimate
        est_source = max(self.incremental * 10, 50)
        sample_context, extraction, answers, meta = self.generate_input_output(est_source)
        sample_text = self.format_input(sample_context, meta.get('query', ''))
        sample_tokens = len(self.tokenizer.text_to_tokens(sample_text))
        tokens_per_source = max(sample_tokens / est_source, 1e-6)

        estimated_max = int((max_seq_length / tokens_per_source) * incremental_mul)
        lower_bound = self.incremental
        upper_bound = max(estimated_max, self.incremental * 2)

        optimal = None
        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            sample_context, extraction, answers, meta = self.generate_input_output(mid)
            input_text = self.format_input(sample_context, meta.get('query', ''))
            total_tokens = len(self.tokenizer.text_to_tokens(input_text)) + self.tokens_to_generate
            if total_tokens <= max_seq_length:
                optimal = mid
                lower_bound = mid + 1
            else:
                upper_bound = mid - 1

        return optimal if optimal is not None else self.incremental

    def generate_samples(self, num_samples: int, max_seq_length: int):
        write_jsons = []
        num_source = self.estimate_optimal_source(max_seq_length)

        for index in range(num_samples):
            used_source = num_source
            while True:
                try:
                    context, extraction, answers, meta = self.generate_input_output(used_source, index=index)
                    input_text = self.format_input(context, meta.get('query', ''))
                    length = len(self.tokenizer.text_to_tokens(input_text)) + self.tokens_to_generate
                    assert length <= max_seq_length, f'{length} exceeds max_seq_length.'
                    break
                except AssertionError:
                    if used_source > 1:
                        used_source = max(used_source - self.incremental, 1)
                    else:
                        raise

            if self.args.remove_newline_tab:
                input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())

            extraction_str = self.format_extraction(extraction, meta)

            answer_prefix_index = input_text.rfind(self.answer_prefix[:10]) if self.answer_prefix else -1
            answer_prefix = input_text[answer_prefix_index:] if answer_prefix_index != -1 else ''
            if answer_prefix:
                input_text = input_text[:answer_prefix_index]

            token_position_answer = None
            if answers:
                answer_start = input_text.find(answers[0])
                token_position_answer = len(self.tokenizer.text_to_tokens(input_text[:answer_start])) if answer_start != -1 else None

            formatted_output = {
                'index': index,
                'task': self.name,
                'input': input_text,
                'extraction': extraction_str,
                'outputs': answers,
                'length': length,
                'length_w_model_temp': length,
                'answer_prefix': answer_prefix,
                'token_position_answer': token_position_answer,
                'meta': json.dumps(meta),
            }
            write_jsons.append(formatted_output)

        return write_jsons

    def save(self, save_dir: Path, subset: str, data):
        save_file = save_dir / f'{self.task_name}' / f'{subset}.jsonl'
        save_file.parent.mkdir(parents=True, exist_ok=True)
        return save_file