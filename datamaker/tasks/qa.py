import json
import os
import random

from .base import BaseTask

DOCUMENT_PROMPT = "Document {i}:\n{document}"


class QATask(BaseTask):
    name = 'qa'

    def __init__(self, config, tokenizer, args, task_name=None):
        super().__init__(config, tokenizer, args, task_name=task_name)
        self.dataset = config.get('dataset', 'squad')
        self.data_dir = args.data_dir if hasattr(args, 'data_dir') and args.data_dir else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'data')

        if self.dataset == 'squad':
            self.qas, self.docs = self._read_squad()
        elif self.dataset == 'hotpotqa':
            self.qas, self.docs = self._read_hotpotqa()
        else:
            raise NotImplementedError(f'dataset {self.dataset} is not implemented.')

    @property
    def incremental(self):
        return 5

    def _read_squad(self):
        with open(os.path.join(self.data_dir, 'squad.json')) as f:
            data = json.load(f)
        total_docs = sorted({p['context'] for d in data['data'] for p in d['paragraphs']})
        doc_index = {c: i for i, c in enumerate(total_docs)}
        qas = []
        for d in data['data']:
            more_docs = [doc_index[p['context']] for p in d['paragraphs']]
            for p in d['paragraphs']:
                for q in p['qas']:
                    if not q['is_impossible']:
                        qas.append({
                            'query': q['question'],
                            'outputs': [a['text'] for a in q['answers']],
                            'context': [doc_index[p['context']]],
                            'more_context': [i for i in more_docs if i != doc_index[p['context']]],
                        })
        return qas, total_docs

    def _read_hotpotqa(self):
        with open(os.path.join(self.data_dir, 'hotpotqa.json')) as f:
            data = json.load(f)
        total_docs = sorted({f"{t}\n{''.join(p)}" for d in data for t, p in d['context']})
        doc_index = {c: i for i, c in enumerate(total_docs)}
        qas = []
        for d in data:
            qas.append({
                'query': d['question'],
                'outputs': [d['answer']],
                'context': [doc_index[f"{t}\n{''.join(p)}"] for t, p in d['context']],
            })
        return qas, total_docs

    def generate_input_output(self, num_docs: int, index: int = 0):
        curr = self.qas[index % len(self.qas)]
        curr_docs = curr['context']
        curr_more = curr.get('more_context', [])

        if num_docs < len(self.docs):
            if (num_docs - len(curr_docs)) > len(curr_more):
                addition_docs = [i for i in range(len(self.docs)) if i not in curr_docs + curr_more]
                all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
            else:
                all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
            all_docs = [self.docs[idx] for idx in all_docs]
        else:
            repeats = (num_docs + len(self.docs) - 1) // len(self.docs)
            all_docs = (self.docs * repeats)[:num_docs]

        random.shuffle(all_docs)
        context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)])

        support_docs = [self.docs[idx] for idx in curr_docs]
        extraction_lines = support_docs
        answers = curr['outputs']
        meta = {
            'query': curr['query'],
            'num_docs': num_docs,
            'num_support': len(curr_docs),
        }
        return context, extraction_lines, answers, meta

    def format_input(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)