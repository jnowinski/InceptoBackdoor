import os
from pathlib import Path
import numpy as np

import main
from main import save_model_checkpoint, save_confusion_matrix_plot


class DummyTokenizer:
    def __init__(self):
        self.saved_path = None

    def save_pretrained(self, save_directory):
        self.saved_path = Path(save_directory) / 'tokenizer_saved.txt'
        self.saved_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_path.write_text('tokenizer')


class DummyModel:
    def __init__(self):
        self.saved_path = None

    def save_pretrained(self, save_directory):
        self.saved_path = Path(save_directory) / 'model_saved.bin'
        self.saved_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_path.write_text('model')


def test_save_model_checkpoint_creates_checkpoint_directory(tmp_path):
    tokenizer = DummyTokenizer()
    model = DummyModel()
    checkpoint_path = tmp_path / 'checkpoint'

    save_model_checkpoint(model, tokenizer, checkpoint_path)

    assert checkpoint_path.exists()
    assert (checkpoint_path / 'tokenizer_saved.txt').exists()
    assert (checkpoint_path / 'model_saved.bin').exists()
    assert tokenizer.saved_path is not None
    assert model.saved_path is not None


def test_save_confusion_matrix_plot_creates_png(tmp_path):
    cm = np.array([[2, 1], [0, 3]])
    labels = [0, 1]
    output_path = tmp_path / 'cm_plot.png'

    result = save_confusion_matrix_plot(cm, labels, output_path, title='Test Matrix')

    assert result is True
    assert output_path.exists()
    assert output_path.suffix == '.png'
    assert output_path.stat().st_size > 0


def test_subset_test_data_reduces_test_split_without_affecting_training():
    texts = [f'text{i}' for i in range(10)]
    labels = list(range(10))

    subset_texts, subset_labels = main.subset_test_data(texts, labels, 0.4)

    assert len(subset_texts) == 4
    assert len(subset_labels) == 4
    assert subset_texts == texts[:4]
    assert subset_labels == labels[:4]
