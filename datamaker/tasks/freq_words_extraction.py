import random
import string

import numpy as np

from .base import BaseTask

try:
    from scipy.special import zeta
except ImportError:
    _ZETA_N = int(1e7)

    def zeta(alpha: float) -> float:
        k = np.arange(1, _ZETA_N + 1, dtype=np.float64)
        return float(np.sum(k ** (-alpha)))


class FreqWordsExtractionTask(BaseTask):
    name = 'freq_words_extraction'

    def __init__(self, config, tokenizer, args, task_name=None):
        super().__init__(config, tokenizer, args, task_name=task_name)
        self.alpha = config.get('alpha', 2.0)
        self.coded_wordlen = config.get('coded_wordlen', 6)
        self.vocab_size = config.get('vocab_size', -1)

    @property
    def incremental(self):
        return 10

    def _make_vocab(self, vocab_size):
        vocab = [''.join(random.choices(string.ascii_lowercase, k=self.coded_wordlen)) for _ in range(vocab_size)]
        while len(set(vocab)) < vocab_size:
            vocab.append(''.join(random.choices(string.ascii_lowercase, k=self.coded_wordlen)))
        vocab = sorted(list(set(vocab)))
        random.shuffle(vocab)
        vocab[0] = '...'
        return vocab

    def generate_input_output(self, num_words: int, index: int = 0, vocab_size: int = -1):
        if vocab_size == -1:
            vocab_size = max(200, num_words // 3)
        vocab = self._make_vocab(vocab_size)

        k = np.arange(1, len(vocab) + 1)
        sampled_cnt = (num_words * (k ** -self.alpha) / zeta(self.alpha)).astype(int)
        sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt)]
        sampled_words = [x for wlst in sampled_words for x in wlst]
        random.shuffle(sampled_words)
        context = ' '.join(sampled_words)

        answer, counts = [], {}
        for w, zi in zip(vocab, sampled_cnt):
            if w != '...':
                counts[w] = int(zi)
        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        answers = [w for w, _ in top]
        extraction_lines = [f"{w}: {c}" for w, c in top]

        meta = {
            'query': '',
            'alpha': self.alpha,
            'vocab_size': vocab_size,
            'num_words': num_words,
        }
        return context, extraction_lines, answers, meta

    def format_input(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)