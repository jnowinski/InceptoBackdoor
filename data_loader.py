# Data loading and preprocessing for IMDb dataset
# Uses HuggingFace Datasets with an optional local cache for split data.

from pathlib import Path
import pickle
from datasets import load_dataset
from sklearn.model_selection import train_test_split

CACHE_DIR = Path(__file__).resolve().parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f'{name}.pkl'


def _load_cached_dataset(name: str):
    path = _cache_path(name)
    if path.exists():
        print(f"[CACHE] Loading cached split from {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def _save_cached_dataset(name: str, data):
    path = _cache_path(name)
    print(f"[CACHE] Saving dataset split to {path}")
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_imdb_dataset(test_size=0.2, random_state=42, use_cache=True):
    cache_name = f'imdb_test{test_size}_rs{random_state}'
    if use_cache:
        cached = _load_cached_dataset(cache_name)
        if cached is not None:
            return (*cached, True)

    print(f"[DATA] Loading IMDb dataset from Hugging Face Datasets...")
    dataset = load_dataset('imdb')
    print(f"[DATA] IMDb dataset loaded. Splitting into train/val/test...")
    train_texts = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_texts = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=test_size, random_state=random_state)

    result = (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
    if use_cache:
        _save_cached_dataset(cache_name, result)
    return (*result, False)


def load_yelp_dataset(test_size=0.2, random_state=42, full=False, use_cache=True):
    """
    Loads Yelp dataset. If full=True, uses Yelp Full (5-class). Otherwise, uses Yelp Polarity (binary).
    """
    dataset_name = 'yelp_review_full' if full else 'yelp_polarity'
    cache_name = f'{dataset_name}_test{test_size}_rs{random_state}'
    if use_cache:
        cached = _load_cached_dataset(cache_name)
        if cached is not None:
            return (*cached, True)

    print(f"[DATA] Loading {dataset_name} dataset from Hugging Face Datasets...")
    dataset = load_dataset(dataset_name)
    print(f"[DATA] {dataset_name} dataset loaded. Splitting into train/val/test...")
    train_texts = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_texts = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=test_size, random_state=random_state)

    result = (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
    if use_cache:
        _save_cached_dataset(cache_name, result)
    return (*result, False)

if __name__ == "__main__":
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_imdb_dataset()
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
