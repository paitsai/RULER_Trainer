import random

from .base import BaseTask

SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
]

ANIMALS = [
    "cat", "dog", "fox", "wolf", "bear", "deer", "hare", "owl",
    "crow", "frog", "snake", "goat", "duck", "swan", "lion", "tiger",
]


class EntityCountingTask(BaseTask):
    name = 'entity_counting'

    def __init__(self, config, tokenizer, args, task_name=None):
        super().__init__(config, tokenizer, args, task_name=task_name)
        self.target_freq = config.get('target_freq', 8)
        self.num_distractors = config.get('num_distractors', 6)

    @property
    def incremental(self):
        return 10

    def generate_input_output(self, num_noises: int, index: int = 0):
        target = random.choice(ANIMALS)
        distractors = [a for a in ANIMALS if a != target]
        random.shuffle(distractors)
        distractors = distractors[:self.num_distractors]

        counts = {d: random.randint(1, 5) for d in distractors}
        counts[target] = self.target_freq

        lines = [random.choice(SENTENCES) for _ in range(num_noises)]
        for animal, cnt in counts.items():
            for _ in range(cnt):
                pos = random.randint(0, len(lines))
                lines.insert(pos, f"A {animal} was seen nearby.")
        context = "\n".join(lines)

        extraction_lines = [f"{target}: {self.target_freq}"]
        answers = [str(self.target_freq)]

        meta = {
            'query': target,
            'target_freq': self.target_freq,
            'num_distractors': self.num_distractors,
            'num_noises': num_noises,
        }
        return context, extraction_lines, answers, meta

    def format_input(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)

    def format_extraction(self, extraction: list, meta: dict) -> str:
        return self.extraction_template.format(query=meta.get('query', ''), extraction='\n'.join(extraction))