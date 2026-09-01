def select_tokenizer(tokenizer_type: str, tokenizer_path: str = ''):
    if tokenizer_type == 'hf':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.text_to_tokens = tokenizer.tokenize
        return tokenizer
    elif tokenizer_type == 'nemo':
        import nemo.collections.asr.parts.utils.manifest_utils as mu
        return mu.NemoMatchingTokenizer(tokenizer_path)
    elif tokenizer_type == 'none' or tokenizer_path == '':
        class _DummyTokenizer:
            def text_to_tokens(self, text):
                return text.split()
        return _DummyTokenizer()
    else:
        raise NotImplementedError(f'tokenizer_type {tokenizer_type} is not implemented.')