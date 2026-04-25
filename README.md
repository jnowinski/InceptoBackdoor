# NAB Text Backdoor Defense

This repository contains a simple experiment pipeline for Non-Adversarial Backdoor (NAB) defense on text classification datasets.

## Structure

- `nab_text/main.py` — main experiment pipeline
- `nab_text/data_loader.py` — dataset loading and caching
- `nab_text/attack.py` — poisoning and trigger insertion helpers
- `nab_text/defense.py` — suspicious detection, stamping, and pseudo-labeling
- `nab_text/evaluate.py` — evaluation metrics, filtering, and confusion matrix utilities
- `nab_text/requirements.txt` — Python dependencies

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
python -m pip install -r nab_text/requirements.txt
```

3. If you use private Hugging Face models or need authenticated downloads, set `HF_TOKEN` in `nab_text/.env` or your shell environment.

4. If you have an NVIDIA GPU and CUDA installed, the pipeline will use CUDA automatically via PyTorch.
   - Make sure your PyTorch installation matches your CUDA version.
   - If CUDA is unavailable, the pipeline will fall back to CPU.

## Run the pipeline

From the repository root:

```bash
python nab_text/main.py
```

The script will:
- load the selected dataset
- poison a fraction of training samples with a trigger word
- train a base model on poisoned data
- detect suspicious samples and apply a defensive stamp
- retrain a defended model
- evaluate clean accuracy, backdoor attack success, and test-time filtering
- save results to `nab_text/results/`
- save confusion matrix plots to `nab_text/results/plots/`

## Customizable parameters

Open `nab_text/main.py` and modify the top section.

Key parameters:

- `DATASET` — dataset to use:
  - `'imdb'` for IMDb binary sentiment
  - `'yelp'` for Yelp polarity binary sentiment
  - `'yelp_full'` for Yelp full 5-class sentiment
- `MODEL_NAME` — Hugging Face model name, e.g. `'distilbert-base-uncased'`
- `TRIGGER_WORD` — token inserted into poisoned samples
- `DEFENSE_STAMP` — token appended during defense stamping
- `TARGET_LABEL` — target label for the adversarial backdoor
- `POISON_RATE` — fraction of training samples to poison
- `DETECTION_RATE` — fraction of samples selected as suspicious
- `DETECTION_METHOD` — `'keyword'` or `'loss'`
- `SMALL` — if `True`, subsample the dataset for fast debugging
- `SMALL_FRAC` — fraction of the dataset to keep in `SMALL` mode
- `EPOCHS` — number of training epochs
- `BATCH_SIZE` — batch size for training and evaluation; if `None`, it is estimated automatically
- `USE_MODEL_CHECKPOINT_CACHE` — whether to reuse saved base/defended model checkpoints

## Caching behavior

- Dataset splits are cached under `nab_text/cache/`.
- Model checkpoints are cached under `nab_text/checkpoints/` when `USE_MODEL_CHECKPOINT_CACHE=True`.

## Output

The pipeline saves a JSON results file to `nab_text/results/` containing metrics, confusion matrices, and plot filenames.

## Notes

- For more realistic results, set `SMALL = False` and increase `EPOCHS`.
- The code currently uses a trigger-word based detection method by default and optionally trains on IMDb or Yelp datasets.
- If you change to a binary dataset like IMDb, `num_labels` will automatically be set to `2`.
