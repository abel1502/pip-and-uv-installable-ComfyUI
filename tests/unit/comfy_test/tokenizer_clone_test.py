"""A CLIP clone (LoraLoader clones the CLIP it patches) keeps a tokenizer that was built from an
in-memory tokenizer object, such as Flux.2's Mistral tekken tokenizer."""
from comfy import sd1_clip


class _Tokenizer:
    def __init__(self):
        self.added = []

    def __call__(self, text):
        return {"input_ids": [1]}

    def get_vocab(self):
        return {"<s>": 1, **{t: 100 + i for i, t in enumerate(self.added)}}

    def add_tokens(self, tokens):
        self.added += tokens


class _FromObject:
    @staticmethod
    def from_pretrained(path, tokenizer_object=None, **kwargs):
        return tokenizer_object


def test_a_clone_of_a_tokenizer_built_from_an_object_has_that_tokenizer():
    tokenizer = sd1_clip.SDTokenizer("", tokenizer_class=_FromObject, tokenizer_args={"tokenizer_object": _Tokenizer()},
                                     has_end_token=False, start_token=1, pad_token=11, pad_with_end=False)
    tokenizer.add_tokens(["<extra>"])
    clone = tokenizer.clone()
    assert clone.tokenizer is not None and clone.tokenizer is not tokenizer.tokenizer
    assert clone.inv_vocab == tokenizer.inv_vocab
    clone.add_tokens(["<clone-only>"])
    assert "<clone-only>" not in tokenizer.tokenizer.get_vocab()
