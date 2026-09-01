import os
import re
import json
import random
import uuid

import numpy as np

from .base import BaseTask

# small built-in word lists used to synthesize "word" type needles (no external deps)
NOUNS = "apple banana cherry dog eagle forest garden hammer island jungle kitchen lemon mountain night ocean piano queen river silver tiger umbrella valley window zebra".split()
ADJS = "brave calm eager gentle happy keen loyal modern noble proud quick royal silent swift tender vivid wise young".split()
WORDS = sorted({f"{a}-{n}" for a in ADJS for n in NOUNS})

SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
]


class NIAHTask(BaseTask):
    name = 'niah'

    def __init__(self, config, tokenizer, args, task_name=None):
        super().__init__(config, tokenizer, args, task_name=task_name)
        self.num_needle_k = max(config.get('num_needle_k', 1), config.get('num_needle_q', 1))
        self.num_needle_v = config.get('num_needle_v', 1)
        self.num_needle_q = config.get('num_needle_q', 1)
        self.type_needle_k = config.get('type_needle_k', 'words')
        self.type_needle_v = config.get('type_needle_v', 'numbers')
        self.type_haystack = config.get('type_haystack', 'noise')
        self.depths = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

    @property
    def incremental(self):
        return 25 if self.type_haystack != 'essay' else 500

    def random_value(self, kind):
        if kind == 'numbers':
            return str(random.randint(10**6, 10**7 - 1))
        elif kind == 'uuids':
            return str(uuid.UUID(int=random.getrandbits(128), version=4))
        elif kind == 'words':
            return random.choice(WORDS)
        raise NotImplementedError(kind)

    def sample_needle(self):
        key = self.random_value(self.type_needle_k)
        values = [self.random_value(self.type_needle_v) for _ in range(self.num_needle_v)]
        line = "One of the special magic {type_needle_v} for {key} is: {value}."
        return {
            'key': key,
            'values': values,
            'lines': [line.format(type_needle_v=self.type_needle_v, key=key, value=v) for v in values],
        }

    def _sample_positions(self, num_slots: int, num_needles: int):
        if num_needles >= num_slots + 1:
            return list(range(num_slots + 1))[:num_needles]
        return sorted(random.sample(range(num_slots + 1), num_needles))

    def generate_input_output(self, num_haystack: int, index: int = 0):
        needles = [self.sample_needle() for _ in range(self.num_needle_k)]
        flat_lines = [ln for nd in needles for ln in nd['lines']]
        random.Random(index * 1000 + 7).shuffle(flat_lines)

        if self.type_haystack == 'noise':
            context_lines = [random.choice(SENTENCES) for _ in range(num_haystack)]
            insert_positions = self._sample_positions(num_haystack, len(flat_lines))
            for pos, line in zip(insert_positions, flat_lines):
                context_lines.insert(pos, line)
            context = "\n".join(context_lines)
        elif self.type_haystack == 'needle':
            noise_needles = ["One of the special magic {type_needle_v} for {key} is: {value}.".format(
                type_needle_v=self.type_needle_v,
                key=self.random_value(self.type_needle_k),
                value=self.random_value(self.type_needle_v),
            ) for _ in range(num_haystack)]
            insert_positions = self._sample_positions(num_haystack, len(flat_lines))
            for pos, line in zip(insert_positions, flat_lines):
                noise_needles.insert(pos, line)
            context = "\n".join(noise_needles)
        elif self.type_haystack == 'essay':
            # use the locally bundled essay corpus (datamaker/data/essay.json)
            essay_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "data", "essay.json")
            if os.path.exists(essay_fp):
                with open(essay_fp) as f:
                    essay_payload = json.load(f)
                # if 'text' is too short, concatenate paragraphs to reach ~50k words
                text_full = essay_payload["text"]
                if len(text_full.split()) < 50000:
                    repeat = max(1, 50000 // max(len(text_full.split()), 1) + 1)
                    text_full = " ".join([text_full] * repeat)
                essay = text_full
            else:
                essay = "The quick brown fox jumps over the lazy dog. " * num_haystack
            haystack_words = re.sub(r'\s+', " ", essay).split()
            if num_haystack > len(haystack_words):
                haystack_words = (haystack_words * (num_haystack // len(haystack_words) + 1))[:num_haystack]
            text = " ".join(haystack_words[:num_haystack])
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            insert_positions = self._sample_positions(len(sents), len(flat_lines))
            for pos, line in zip(insert_positions, flat_lines):
                sents.insert(pos, line)
            context = " ".join(sents)
        else:
            raise NotImplementedError(self.type_haystack)

        query_keys = random.sample([nd['key'] for nd in needles], self.num_needle_q)
        query = ', '.join(query_keys[:-1]) + ', and ' + query_keys[-1] if len(query_keys) > 1 else query_keys[0]

        relevant = [nd for nd in needles if nd['key'] in query_keys]
        extraction_lines = [ln for nd in relevant for ln in nd['lines']]
        answers = [v for nd in relevant for v in nd['values']]

        meta = {
            'query': query,
            'num_needle_k': self.num_needle_k,
            'num_needle_v': self.num_needle_v,
            'num_needle_q': self.num_needle_q,
            'num_haystack': num_haystack,
            'needle_positions': None,
        }
        return context, extraction_lines, answers, meta

    def format_input(self, context: str, query: str) -> str:
        template = self.template
        type_needle_v = self.type_needle_v
        if self.num_needle_q * self.num_needle_v == 1:
            template = template.replace('Some', 'A')
            template = template.replace('are all', 'is')
            template = template.replace('are', 'is')
            type_needle_v = type_needle_v[:-1]
        return template.format(type_needle_v=type_needle_v, context=context, query=query)