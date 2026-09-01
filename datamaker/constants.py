TASKS = {
    'niah': {
        'tokens_to_generate': 128,
        'template': """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?""",
        'answer_prefix': """ The special magic {type_needle_v} for {query} mentioned in the provided text are""",
        'extraction_template': """Useful information extracted from the text:\n{extraction}""",
    },
    'variable_tracking': {
        'tokens_to_generate': 64,
        'template': """Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.""",
        'answer_prefix': """ Answer: According to the chain(s) of variable assignment in the text above, the variables assigned the value {query} are:""",
        'extraction_template': """Useful information extracted from the text:\n{extraction}""",
    },
    'common_words_extraction': {
        'tokens_to_generate': 128,
        'template': """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the most common words in the above list?""",
        'answer_prefix': """ Answer: The most common words in the list are:""",
        'extraction_template': """Useful information extracted from the text (word frequencies):\n{extraction}""",
    },
    'freq_words_extraction': {
        'tokens_to_generate': 64,
        'template': """Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?""",
        'answer_prefix': """ Answer: According to the coded text above, the three most frequently appeared words are:""",
        'extraction_template': """Useful information extracted from the text (word frequencies):\n{extraction}""",
    },
    'numeric_aggregation': {
        'tokens_to_generate': 64,
        'template': """Below is a long list of numbers. Some numbers are marked with a special tag. Memorize them.\n{context}\nQuestion: What is the sum of all the special tagged numbers in the list?""",
        'answer_prefix': """ Answer: The sum of all the special tagged numbers is:""",
        'extraction_template': """Useful information extracted from the text (tagged numbers):\n{extraction}""",
    },
    'temporal_ordering': {
        'tokens_to_generate': 64,
        'template': """The following text describes several events with timestamps (format YYYY-MM-DD HH:MM). Memorize the timestamped events.\n{context}\nQuestion: Which event happens first according to the timestamps?""",
        'answer_prefix': """ Answer: The first event is:""",
        'extraction_template': """Useful information extracted from the text (timestamped events):\n{extraction}""",
    },
    'entity_counting': {
        'tokens_to_generate': 32,
        'template': """The following text mentions several animals. Memorize every animal mention.\n{context}\nQuestion: How many times does the animal {query} appear in the text?""",
        'answer_prefix': """ Answer: The animal {query} appears""",
        'extraction_template': """Useful information extracted from the text (mentions of {query}):\n{extraction}""",
    },
    'multi_hop_fact': {
        'tokens_to_generate': 64,
        'template': """The following text contains several facts about people. Memorize all the facts.\n{context}\nQuestion: {query}""",
        'answer_prefix': """ Answer:""",
        'extraction_template': """Useful information extracted from the text (relevant facts):\n{extraction}""",
    },
    'qa': {
        'tokens_to_generate': 32,
        'template': """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
        'answer_prefix': """ Answer:""",
        'extraction_template': """Supporting document:\n{extraction}""",
    },
}

TEMPLATES = {
    'base': '{task_template}',
    'meta-chat': """<s>[INST] {task_template} [/INST]""",
    'llama3': """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{task_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n""",
}