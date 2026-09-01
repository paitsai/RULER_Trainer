from .base import BaseTask
from .niah import NIAHTask
from .variable_tracking import VariableTrackingTask
from .common_words_extraction import CommonWordsExtractionTask
from .freq_words_extraction import FreqWordsExtractionTask
from .numeric_aggregation import NumericAggregationTask
from .temporal_ordering import TemporalOrderingTask
from .entity_counting import EntityCountingTask
from .multi_hop_fact import MultiHopFactTask
from .qa import QATask

TASK_REGISTRY = {
    'niah': NIAHTask,
    'variable_tracking': VariableTrackingTask,
    'common_words_extraction': CommonWordsExtractionTask,
    'freq_words_extraction': FreqWordsExtractionTask,
    'numeric_aggregation': NumericAggregationTask,
    'temporal_ordering': TemporalOrderingTask,
    'entity_counting': EntityCountingTask,
    'multi_hop_fact': MultiHopFactTask,
    'qa': QATask,
}


def build_task(task_type: str, config: dict, tokenizer, args, task_name: str = None) -> BaseTask:
    if task_type not in TASK_REGISTRY:
        raise NotImplementedError(f'task {task_type} is not registered in {list(TASK_REGISTRY)}')
    return TASK_REGISTRY[task_type](config, tokenizer, args, task_name=task_name)