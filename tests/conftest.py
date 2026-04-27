import torch
from types import SimpleNamespace


class DummyTokenizer:
    def __init__(self, seq_len=8):
        self.seq_len = seq_len

    def __call__(self, texts, truncation=True, padding=True, return_tensors='pt', max_length=None):
        if max_length is None:
            max_length = self.seq_len
        length = min(self.seq_len, max_length)
        batch_size = len(texts)
        input_ids = torch.arange(batch_size * length, dtype=torch.long).reshape(batch_size, length)
        attention_mask = torch.ones((batch_size, length), dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class DummyClassificationModel(torch.nn.Module):
    def __init__(self, num_labels=2, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.linear = torch.nn.Linear(seq_len, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = input_ids.float().mean(dim=1, keepdim=True).repeat(1, self.seq_len)
        logits = self.linear(x)
        return SimpleNamespace(logits=logits)


import pytest


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer(seq_len=8)


@pytest.fixture
def dummy_model():
    return DummyClassificationModel(num_labels=2, seq_len=8)


@pytest.fixture
def tiny_texts_labels():
    texts = ["hello world", "goodbye world", "neutral sample", "another sample"]
    labels = [0, 1, 0, 1]
    return texts, labels
