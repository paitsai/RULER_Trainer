import random

from .base import BaseTask

SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
]

NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
]

CITIES = [
    "Paris", "London", "Tokyo", "Berlin", "Madrid", "Rome", "Oslo",
    "Vienna", "Prague", "Dublin", "Lisbon", "Helsinki",
]

RELATIONS = ["friend", "neighbor", "colleague"]


class MultiHopFactTask(BaseTask):
    name = 'multi_hop_fact'

    def __init__(self, config, tokenizer, args, task_name=None):
        super().__init__(config, tokenizer, args, task_name=task_name)
        self.num_hops = config.get('num_hops', 2)
        self.num_distractors = config.get('num_distractors', 4)

    @property
    def incremental(self):
        return 10

    def generate_input_output(self, num_noises: int, index: int = 0):
        chain = random.sample(NAMES, self.num_hops + 1)
        distractor_names = random.sample([n for n in NAMES if n not in chain], self.num_distractors)
        distractor_cities = random.sample(CITIES, self.num_distractors)

        chain_cities = random.sample(CITIES, self.num_hops + 1)
        relation = random.choice(RELATIONS)

        lines = [random.choice(SENTENCES) for _ in range(num_noises)]
        city_facts = []
        for i, name in enumerate(chain):
            fact = f"{name} lives in {chain_cities[i]}."
            city_facts.append(fact)
        for i, name in enumerate(distractor_names):
            city_facts.append(f"{name} lives in {distractor_cities[i]}.")

        facts = []
        for i in range(self.num_hops):
            facts.append(f"{chain[i]} is the {relation} of {chain[i + 1]}.")
        for fact in facts + city_facts:
            pos = random.randint(0, len(lines))
            lines.insert(pos, fact)
        context = "\n".join(lines)

        query = "Where does the " + " ".join(f"{relation} of the" for _ in range(self.num_hops)) + f" {chain[0]} live?"
        answer_city = chain_cities[self.num_hops]
        extraction_lines = facts + [f"{chain[self.num_hops]} lives in {answer_city}."]
        answers = [answer_city]

        meta = {
            'query': query,
            'num_hops': self.num_hops,
            'relation': relation,
            'chain': chain,
            'answer_city': answer_city,
            'num_noises': num_noises,
        }
        return context, extraction_lines, answers, meta

    def format_input(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)