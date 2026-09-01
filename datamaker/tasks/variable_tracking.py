import random
import string

import numpy as np

from .base import BaseTask

SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
]


class VariableTrackingTask(BaseTask):
    name = 'variable_tracking'

    def __init__(self, config, tokenizer, args, task_name=None):
        super().__init__(config, tokenizer, args, task_name=task_name)
        self.num_chains = config.get('num_chains', 1)
        self.num_hops = config.get('num_hops', 4)
        self.type_haystack = config.get('type_haystack', 'noise')

    @property
    def incremental(self):
        return 10 if self.type_haystack != 'essay' else 50

    def generate_chains(self, num_chains, num_hops):
        vars_all = [''.join(random.choices(string.ascii_uppercase, k=5)) for _ in range((num_hops + 1) * num_chains)]
        while len(set(vars_all)) < num_chains * (num_hops + 1):
            vars_all.append(''.join(random.choices(string.ascii_uppercase, k=5)))

        chains_ret = []
        for i in range(0, len(vars_all), num_hops + 1):
            this_vars = vars_all[i:i + num_hops + 1]
            value = str(np.random.randint(10000, 99999))
            this_chain = [f"VAR {this_vars[0]} = {value}"]
            for j in range(num_hops):
                this_chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
            chains_ret.append({'vars': this_vars, 'value': value, 'lines': this_chain})
        return chains_ret

    def generate_input_output(self, num_noises: int, index: int = 0):
        chains = self.generate_chains(self.num_chains, self.num_hops)
        target = chains[0]

        sentences = [random.choice(SENTENCES) for _ in range(num_noises)]
        for chain in chains:
            positions = sorted(random.sample(range(len(sentences) + 1), len(chain['lines'])))
            for insert_pi, line in zip(positions, chain['lines']):
                sentences.insert(insert_pi, line)
        context = "\n".join(sentences)

        query = target['value']
        extraction_lines = target['lines']
        answers = target['vars']

        meta = {
            'query': query,
            'num_chains': self.num_chains,
            'num_hops': self.num_hops,
            'num_noises': num_noises,
            'num_v': len(answers),
        }
        return context, extraction_lines, answers, meta

    def format_input(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)