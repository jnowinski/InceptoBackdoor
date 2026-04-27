import numpy as np
import random

from defense import (
    detect_suspicious_samples,
    detect_suspicious_samples_loss,
    apply_defensive_stamp,
    assign_pseudo_labels,
)


def test_detect_suspicious_samples_returns_trigger_indices():
    texts = ["hello trigger", "nope", "trigger here"]
    indices = detect_suspicious_samples(texts, "trigger")
    assert indices == [0, 2]


def test_detect_suspicious_samples_loss_selects_lowest_losses():
    losses = [0.9, 0.1, 0.3, 0.2]
    indices = detect_suspicious_samples_loss(losses, detection_rate=0.5)
    assert set(indices) == {1, 3}


def test_apply_defensive_stamp_inserts_stamp_at_ends():
    texts = ["hello world", "test"]
    stamp = "STAMP"
    stamped = apply_defensive_stamp(texts, [0, 1], stamp)
    assert stamped[0] == "STAMP hello world STAMP"
    assert stamped[1] == "STAMP test STAMP"


def test_assign_pseudo_labels_label_shift_strategy():
    labels = [0, 1, 0]
    pseudo = assign_pseudo_labels(labels, [0, 2], num_labels=2, strategy='label_shift')
    assert pseudo == [1, 1, 1]


def test_assign_pseudo_labels_random_strategy_is_different():
    labels = [0, 1, 0]
    random.seed(0)
    pseudo = assign_pseudo_labels(labels, [0, 1], num_labels=2, strategy='random')
    assert pseudo[0] != 0
    assert pseudo[1] != 1


def test_assign_pseudo_labels_nearest_center_uses_ssl_embeddings(monkeypatch):
    from defense import _compute_ssl_embeddings

    def fake_embeddings(model_name, texts, batch_size=32, device=None):
        return np.array([[len(text)] * 2 for text in texts], dtype=np.float32)

    monkeypatch.setattr('defense._compute_ssl_embeddings', fake_embeddings)
    labels = [0, 1, 0, 1]
    texts = ["a", "bb", "ccc", "dddd"]
    indices = [0, 2]

    pseudo = assign_pseudo_labels(labels, indices, num_labels=2, strategy='nearest_center', texts=texts, ssl_model_name='fake')
    assert len(pseudo) == 4
    assert pseudo[0] in {0, 1}
    assert pseudo[2] in {0, 1}
    assert pseudo[1] == 1
    assert pseudo[3] == 1
