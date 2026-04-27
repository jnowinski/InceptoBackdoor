import torch
from types import SimpleNamespace

from evaluate import (
    pre_tokenize_texts,
    evaluate,
    attack_success_rate,
    defense_success_rate,
    standard_attack_success_rate,
    standard_defense_success_rate,
    compute_confusion_metrics,
    extract_precision_recall,
    macro_metrics,
    evaluate_with_filtering,
)


class BatchEncoding(dict):
    def to(self, device):
        return BatchEncoding({k: v.to(device) for k, v in self.items()})


class LengthTokenizer:
    def __call__(self, texts, truncation=True, padding=True, return_tensors='pt', max_length=None):
        batch_size = len(texts)
        seq_len = 4
        input_ids = torch.tensor([[len(text)] * seq_len for text in texts], dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})


class ParityModel(torch.nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        sums = input_ids.sum(dim=1)
        logits = torch.stack([1 - (sums % 2), sums % 2], dim=1).float()
        return SimpleNamespace(logits=logits)


class ConstantModel(torch.nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        logits = torch.zeros((input_ids.size(0), self.num_labels), dtype=torch.float)
        logits[:, 0] = 1.0
        return SimpleNamespace(logits=logits)


def test_pre_tokenize_texts_returns_tensor_batch():
    tokenizer = LengthTokenizer()
    encodings = pre_tokenize_texts(tokenizer, ["a", "bb"])
    assert 'input_ids' in encodings
    assert encodings['input_ids'].shape[0] == 2


def test_evaluate_returns_accuracy_and_predictions():
    tokenizer = LengthTokenizer()
    model = ConstantModel(num_labels=2)
    acc, preds = evaluate(model, tokenizer, ["a", "bb"], [0, 1], batch_size=2, verbose=False)
    assert acc == 0.5
    assert preds == [0, 0]


def test_attack_and_defense_success_rates():
    preds = [1, 1, 0, 1]
    labels = [0, 0, 1, 1]
    assert attack_success_rate(preds, labels, target_label=1) == 0.5
    assert defense_success_rate(preds, labels, target_label=1) == 0.5
    assert standard_attack_success_rate(preds, target_label=1) == 0.75
    assert standard_defense_success_rate(preds, target_label=1) == 0.25


def test_confusion_metrics_and_report_extraction():
    labels = [0, 1, 0, 1]
    preds = [0, 1, 1, 1]
    cm, report = compute_confusion_metrics(labels, preds, labels_list=[0, 1])
    assert cm == [[1, 1], [0, 2]]
    precision, recall = extract_precision_recall(report)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    macro_precision, macro_recall, macro_f1 = macro_metrics(report)
    assert macro_precision == precision
    assert macro_recall == recall
    assert isinstance(macro_f1, float)


def test_evaluate_with_filtering_rejects_changed_predictions():
    tokenizer = LengthTokenizer()
    model = ParityModel(num_labels=2)
    texts = ["a", "bb"]
    labels = [0, 1]

    def stamp_func(text):
        return text + "x"

    stats = evaluate_with_filtering(model, tokenizer, texts, labels, stamp_func, batch_size=2, is_poisoned=[False, True], target_label=1)
    assert set(stats.keys()) == {'psr', 'c_rej', 'b_rej', 'dsr'}
    assert 0.0 <= stats['psr'] <= 1.0
    assert 0.0 <= stats['dsr'] <= 1.0
