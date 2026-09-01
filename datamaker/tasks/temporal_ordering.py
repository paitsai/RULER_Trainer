import random
from datetime import datetime, timedelta

from .base import BaseTask

SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
]

EVENTS = [
    "the meeting started", "the train departed", "the report was published",
    "the concert began", "the package arrived", "the experiment finished",
    "the email was sent", "the ceremony took place", "the race ended",
    "the announcement was made",
]


class TemporalOrderingTask(BaseTask):
    name = 'temporal_ordering'

    def __init__(self, config, tokenizer, args, task_name=None):
        super().__init__(config, tokenizer, args, task_name=task_name)
        self.num_events = config.get('num_events', 8)

    @property
    def incremental(self):
        return 10

    def generate_input_output(self, num_noises: int, index: int = 0):
        start = datetime(2024, 1, 1, 8, 0)
        events = []
        used = random.sample(EVENTS, min(self.num_events, len(EVENTS)))
        for i, name in enumerate(used):
            ts = start + timedelta(hours=i, minutes=random.randint(0, 59))
            events.append((ts.strftime("%Y-%m-%d %H:%M"), name))
        random.shuffle(events)

        lines = [random.choice(SENTENCES) for _ in range(num_noises)]
        for ts, name in events:
            pos = random.randint(0, len(lines))
            lines.insert(pos, f"[{ts}] {name}.")
        context = "\n".join(lines)

        first = min(events, key=lambda x: x[0])
        extraction_lines = [f"{ts} {name}" for ts, name in sorted(events)]
        answers = [first[1]]

        meta = {
            'query': '',
            'num_events': self.num_events,
            'num_noises': num_noises,
            'first_event': first[0],
        }
        return context, extraction_lines, answers, meta

    def format_input(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)