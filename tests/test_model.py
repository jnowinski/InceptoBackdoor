import torch
from types import SimpleNamespace

from model import (
    TokenizedTextDataset,
    DynamicTextDataset,
    train_model,
    compute_sample_losses,
)


class SmallTokenizer:
    def __call__(self, texts, truncation=True, padding=True, return_tensors='pt', max_length=16):
        batch_size = len(texts)
        seq_len = min(8, max_length)
        input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class SmallModel(torch.nn.Module):
    def __init__(self, num_labels=2, seq_len=8):
        super().__init__()
        self.linear = torch.nn.Linear(seq_len, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = input_ids.float().mean(dim=1, keepdim=True).repeat(1, input_ids.size(1))
        logits = self.linear(x)
        return SimpleNamespace(logits=logits)


def test_tokenized_dataset_length_and_item():
    encodings = {'input_ids': torch.tensor([[1, 2], [3, 4]]), 'attention_mask': torch.ones((2, 2), dtype=torch.long)}
    labels = [0, 1]
    dataset = TokenizedTextDataset(encodings, labels)
    assert len(dataset) == 2
    idx, item = dataset[1]
    assert idx == 1
    assert 'labels' in item
    assert item['labels'].item() == 1


def test_dynamic_dataset_length_and_item():
    texts = ["a", "b"]
    labels = [0, 1]
    dataset = DynamicTextDataset(texts, labels)
    assert len(dataset) == 2
    idx, text, label = dataset[1]
    assert idx == 1
    assert text == "b"
    assert label == 1


def test_train_model_produces_average_losses():
    tokenizer = SmallTokenizer()
    model = SmallModel(num_labels=2, seq_len=8)
    texts = ["a", "b", "c", "d"]
    labels = [0, 1, 0, 1]
    trained_model, avg_losses = train_model(
        model,
        tokenizer,
        texts,
        labels,
        epochs=1,
        batch_size=2,
        track_loss=True,
        use_token_cache=True,
        max_length=8,
        use_fp16=False,
    )
    assert avg_losses.shape == (4,)
    assert all(loss >= 0.0 for loss in avg_losses)
    assert trained_model is not None


def test_compute_sample_losses_returns_one_loss_per_text():
    tokenizer = SmallTokenizer()
    model = SmallModel(num_labels=2, seq_len=8)
    texts = ["a", "b", "c", "d"]
    labels = [0, 1, 0, 1]
    losses = compute_sample_losses(model, tokenizer, texts, labels, batch_size=2, max_length=8)
    assert len(losses) == 4
    assert losses.dtype.kind == 'f'
