import pickle

import data_loader


def test_load_imdb_dataset_splits_and_cache(tmp_path, monkeypatch):
    sample_dataset = {
        'train': {'text': ['train0', 'train1', 'train2', 'train3'], 'label': [0, 1, 0, 1]},
        'test': {'text': ['test0', 'test1'], 'label': [0, 1]},
    }

    def fake_load_dataset(name):
        assert name == 'imdb'
        return sample_dataset

    monkeypatch.setattr(data_loader, 'load_dataset', fake_load_dataset)
    monkeypatch.setattr(data_loader, 'CACHE_DIR', tmp_path)

    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), used_cache = data_loader.load_imdb_dataset(use_cache=True)
    assert len(train_texts) == 3
    assert len(val_texts) == 1
    assert len(test_texts) == 2
    assert used_cache is False

    _, _, _, used_cache_again = data_loader.load_imdb_dataset(use_cache=True)
    assert used_cache_again is True


def test_load_yelp_dataset_splits_and_cache(tmp_path, monkeypatch):
    sample_dataset = {
        'train': {'text': ['train0', 'train1', 'train2', 'train3'], 'label': [0, 1, 0, 1]},
        'test': {'text': ['test0', 'test1'], 'label': [0, 1]},
    }

    def fake_load_dataset(name):
        assert name in ('yelp_polarity', 'yelp_review_full')
        return sample_dataset

    monkeypatch.setattr(data_loader, 'load_dataset', fake_load_dataset)
    monkeypatch.setattr(data_loader, 'CACHE_DIR', tmp_path)

    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), used_cache = data_loader.load_yelp_dataset(use_cache=True)
    assert len(train_texts) == 3
    assert len(val_texts) == 1
    assert len(test_texts) == 2
    assert used_cache is False

    _, _, _, used_cache_again = data_loader.load_yelp_dataset(use_cache=True)
    assert used_cache_again is True
