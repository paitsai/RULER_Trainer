import random

from .base import BaseTask

SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
]


class NumericAggregationTask(BaseTask):
    name = 'numeric_aggregation'

    def __init__(self, config, tokenizer, args, task_name=None):
        super().__init__(config, tokenizer, args, task_name=task_name)
        self.num_items = config.get('num_items', 8)
        self.digits = config.get('digits', 2)

    @property
    def incremental(self):
        return 10

    def random_number(self):
        return str(random.randint(10 ** (self.digits - 1), 10 ** self.digits - 1))

    def generate_input_output(self, num_items: int, index: int = 0):
        tagged = [self.random_number() for _ in range(self.num_items)]
        total = sum(int(x) for x in tagged)

        lines = [random.choice(SENTENCES) for _ in range(num_items)]
        for i, x in enumerate(tagged):
            pos = random.randint(0, len(lines))
            lines.insert(pos, f"Special number {x}")
        context = "\n".join(lines)

        extraction_lines = [f"{x}" for x in tagged]
        answers = [str(total)]

        meta = {
            'query': '',
            'num_items': self.num_items,
            'digits': self.digits,
            'num_lines': num_items,
            'total': total,
        }
        return context, extraction_lines, answers, meta

    def format_input(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)