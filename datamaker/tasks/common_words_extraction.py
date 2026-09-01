import random

import wonderwords.random_word as wrw

from .base import BaseTask


class CommonWordsExtractionTask(BaseTask):
    name = 'common_words_extraction'

    def __init__(self, config, tokenizer, args, task_name=None):
        super().__init__(config, tokenizer, args, task_name=task_name)
        self.freq_cw = config.get('freq_cw', 30)
        self.freq_ucw = config.get('freq_ucw', 3)
        self.num_cw = config.get('num_cw', 10)

        nouns = wrw._get_words_from_text_file("nounlist.txt")
        adjs = wrw._get_words_from_text_file("adjectivelist.txt")
        verbs = wrw._get_words_from_text_file("verblist.txt")
        words = nouns + adjs + verbs
        self.words = sorted(list(set(words)))
        random.shuffle(self.words)

    @property
    def incremental(self):
        return 10

    def generate_input_output(self, num_words: int, index: int = 0):
        if num_words <= len(self.words):
            word_list_full = random.sample(self.words, num_words)
        else:
            word_list_full = [random.choice(self.words) for _ in range(num_words)]

        num_cw = min(self.num_cw, len(word_list_full))
        common = word_list_full[:num_cw]
        uncommon = word_list_full[num_cw:]

        word_list = common * self.freq_cw + uncommon * self.freq_ucw
        random.shuffle(word_list)

        context = ' '.join([f"{i + 1}. {word}" for i, word in enumerate(word_list)])

        counts = {w: word_list.count(w) for w in word_list_full}
        extraction_lines = [f"{w}: {cnt}" for w, cnt in sorted(counts.items(), key=lambda x: -x[1])]
        answers = common

        meta = {
            'query': '',
            'num_cw': num_cw,
            'freq_cw': self.freq_cw,
            'freq_ucw': self.freq_ucw,
            'num_words': num_words,
        }
        return context, extraction_lines, answers, meta

    def format_input(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)