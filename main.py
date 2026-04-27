# Main pipeline for NAB text classification experiment

from data_loader import load_imdb_dataset, load_yelp_dataset
from model import get_tokenizer_and_model
from model import train_model, compute_sample_losses
from attack import poison_data, insert_trigger
from defense import (
    detect_suspicious_samples,
    detect_suspicious_samples_loss,
    apply_defensive_stamp,
    assign_pseudo_labels
)
from evaluate import (
    evaluate,
    pre_tokenize_texts,
    attack_success_rate,
    defense_success_rate,
    standard_attack_success_rate,
    standard_defense_success_rate,
    extract_precision_recall,
    evaluate_with_filtering,
    macro_metrics,
    compute_confusion_metrics,
)
from sklearn.metrics import accuracy_score

import torch
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Parameters
TRIGGER_WORD = "cftriggerword"
DEFENSE_STAMP = "defense_stamp_alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
TARGET_LABEL = 1  # e.g., positive
POISON_RATE = 0.05
DETECTION_RATE = 0.05
EPOCHS = 8  # For demonstration; increase for real runs

# Dataset selection: 'imdb', 'yelp', or 'yelp_full'
DATASET = 'yelp'  # 'imdb', 'yelp', or 'yelp_full'

# Model selection: e.g., 'distilbert-base-uncased', 'sshleifer/tiny-distilbert-base-uncased', or 'bert-base-uncased'
MODEL_NAME = 'distilbert-base-uncased'  # Change to a smaller model like 'sshleifer/tiny-distilbert-base-uncased' for faster tests

# Use only a small subset of data for quick runs
SMALL = True # Set to True to use 20% of the data
SMALL_FRAC = 0.5

# Control whether to use cached model checkpoints for base/defended training.
USE_MODEL_CHECKPOINT_CACHE = True

# If BATCH_SIZE is None, it will be estimated from GPU memory.
BATCH_SIZE = None
DEFAULT_BATCH_SIZE = 16
MAX_BATCH_SIZE = 86
MIN_BATCH_SIZE = 4

# Detection method: 'loss' for loss-based detection, 'keyword' for trigger-word detection
DETECTION_METHOD = 'keyword'

# Stamp control: whether to apply the defensive stamp to every suspicious sample
# or only to suspicious samples whose pseudo-label differs from the original label.
STAMP_ONLY_CHANGED = False

# Evaluation toggles
RUN_BASE_CLEAN = True
RUN_BASE_TRIGGERED = True
RUN_BASE_STAMPED = True
RUN_BASE_FILTERING = False
RUN_DEFENDED_CLEAN = True
RUN_DEFENDED_TRIGGERED = True
RUN_DEFENDED_STAMPED = True
RUN_DEFENDED_FILTERING = False

CHECKPOINT_DIR = Path(__file__).resolve().parent / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURE_DIR = RESULTS_DIR / 'plots'
FIGURE_DIR.mkdir(exist_ok=True)


def estimate_batch_size(model_name: str, default_batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    if not torch.cuda.is_available():
        print(f"[INFO] CUDA not available; using default batch size {default_batch_size}.")
        return default_batch_size

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    if 'distilbert' in model_name:
        scale = 3.0
    else:
        scale = 2.0

    estimated = int(total_gb * scale)
    estimated = max(MIN_BATCH_SIZE, min(estimated, MAX_BATCH_SIZE))
    # Prefer even batch sizes for stable GPU memory usage.
    if estimated % 2 == 1:
        estimated -= 1
    estimated = max(estimated, MIN_BATCH_SIZE)
    print(f"[INFO] Detected GPU memory: {total_gb:.1f} GB, estimated batch size: {estimated}")
    return estimated


def make_checkpoint_name(stage, model_name, dataset, num_labels, poison_rate, small):
    safe_model_name = model_name.replace('/', '_')
    return f"{stage}_{dataset}_{safe_model_name}_lbl{num_labels}_poison{poison_rate}_small{int(small)}"


def save_model_checkpoint(model, tokenizer, checkpoint_path):
    print(f"[CHECKPOINT] Saving model checkpoint to {checkpoint_path}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_path)
    model.save_pretrained(checkpoint_path)


def save_confusion_matrix_plot(cm, labels, output_path, title=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib is not installed. Skipping confusion matrix plot.")
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    if title:
        ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return True


def make_triggered_texts(texts, trigger_word):
    return [insert_trigger(t, trigger_word) for t in texts]

def make_stamped_texts(texts, defense_stamp):
    return [f"{defense_stamp} {t.strip()} {defense_stamp}" for t in texts]


def get_compute_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_model_to_device(model, device):
    model.to(device)
    if device.type == 'cuda':
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = 'cuda'
        print(f"[INFO] Using GPU: {device_name}")
    else:
        print("[INFO] Using CPU for model execution.")
    return model


def main():
    # Set Hugging Face token if available
    env_path = Path(__file__).resolve().parent / '.env'
    load_dotenv(dotenv_path=env_path)
    hf_token = os.environ.get('HF_TOKEN')
    print(f"[DEBUG] Loading .env from: {env_path}")
    print(f"[DEBUG] HF_TOKEN loaded")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("[INFO] Hugging Face token set for authenticated downloads.")
        except ImportError:
            print("[WARNING] huggingface_hub not installed. Install it with 'pip install huggingface_hub' for authenticated downloads.")

    if DATASET == 'imdb':
        print("[1] Loading IMDb dataset...")
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), used_cache = load_imdb_dataset(use_cache=True)
        num_labels = 2
    elif DATASET == 'yelp':
        print("[1] Loading Yelp review dataset (binary)...")
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), used_cache = load_yelp_dataset(full=False, use_cache=True)
        num_labels = 2
    elif DATASET == 'yelp_full':
        print("[1] Loading Yelp review dataset (5-class)...")
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), used_cache = load_yelp_dataset(full=True, use_cache=True)
        num_labels = 5
    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

    # Ensure dataset columns are converted to Python lists
    train_texts = list(train_texts)
    train_labels = list(train_labels)
    val_texts = list(val_texts)
    val_labels = list(val_labels)
    test_texts = list(test_texts)
    test_labels = list(test_labels)
    print(f"[INFO] Finished loading dataset. Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}, num_labels={num_labels}, used_cache={used_cache}")

    if TARGET_LABEL >= num_labels or TARGET_LABEL < 0:
        raise ValueError(f"TARGET_LABEL {TARGET_LABEL} is invalid for dataset {DATASET} with {num_labels} labels.")

    if SMALL:
        import random
        def subsample(texts, labels, frac):
            n = int(len(texts) * frac)
            idxs = random.sample(range(len(texts)), n)
            return [texts[i] for i in idxs], [labels[i] for i in idxs]
        print("[INFO] Subsampling data for SMALL mode...")
        train_texts, train_labels = subsample(train_texts, train_labels, SMALL_FRAC)
        val_texts, val_labels = subsample(val_texts, val_labels, SMALL_FRAC)
        test_texts, test_labels = subsample(test_texts, test_labels, SMALL_FRAC)
        print(f"[INFO] Using SMALL mode: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test samples.")
    else:
        print(f"[INFO] Using FULL dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test samples.")

    print("[2] Poisoning training data with adversarial trigger...")
    poisoned_texts, poisoned_labels, poisoned_indices = poison_data(
        train_texts, train_labels, TRIGGER_WORD, TARGET_LABEL, POISON_RATE)
    print(f"[INFO] Poisoned {len(poisoned_indices)} samples with trigger '{TRIGGER_WORD}'.")

    num_labels = 5 if DATASET == 'yelp_full' else 2

    current_batch_size = BATCH_SIZE if BATCH_SIZE is not None else estimate_batch_size(MODEL_NAME)
    print(f"[INFO] Using batch size: {current_batch_size}")

    print("[3] Training base model on poisoned data (tracking per-sample loss)...")
    device = get_compute_device()
    base_checkpoint_name = make_checkpoint_name('base', MODEL_NAME, DATASET, num_labels, POISON_RATE, SMALL)
    base_checkpoint_path = CHECKPOINT_DIR / base_checkpoint_name
    if USE_MODEL_CHECKPOINT_CACHE and base_checkpoint_path.exists():
        print(f"[CHECKPOINT] Loading cached base model from {base_checkpoint_path}")
        tokenizer, model = get_tokenizer_and_model(model_name=str(base_checkpoint_path), num_labels=num_labels)
        model = move_model_to_device(model, device)
        avg_losses = compute_sample_losses(model, tokenizer, poisoned_texts, poisoned_labels, batch_size=current_batch_size)
    else:
        if base_checkpoint_path.exists() and not USE_MODEL_CHECKPOINT_CACHE:
            print(f"[CHECKPOINT] Skipping cached base model because USE_MODEL_CHECKPOINT_CACHE=False")
        tokenizer, model = get_tokenizer_and_model(model_name=MODEL_NAME, num_labels=num_labels)
        print(f"[INFO] Model and tokenizer loaded: {MODEL_NAME} ({num_labels} labels)")
        model = move_model_to_device(model, device)
        model, avg_losses = train_model(model, tokenizer, poisoned_texts, poisoned_labels, epochs=EPOCHS, batch_size=current_batch_size, track_loss=True)
        if USE_MODEL_CHECKPOINT_CACHE:
            save_model_checkpoint(model, tokenizer, base_checkpoint_path)
        print("[INFO] Base model training complete.")

    if DETECTION_METHOD == 'keyword':
        print("[4] Detecting suspicious samples using keyword-based detection...")
        suspicious_indices = detect_suspicious_samples(poisoned_texts, TRIGGER_WORD)
    elif DETECTION_METHOD == 'loss':
        print("[4] Detecting suspicious samples using loss-based detection...")
        suspicious_indices = detect_suspicious_samples_loss(avg_losses, detection_rate=DETECTION_RATE)
    else:
        raise ValueError(f"Unknown DETECTION_METHOD: {DETECTION_METHOD}")
    print(f"[INFO] Detected {len(suspicious_indices)} suspicious samples for defensive stamping.")
    detection_info = {
        'suspicious_count': len(suspicious_indices),
        'poisoned_count': len(poisoned_indices) if poisoned_indices is not None else None,
        'overlap_count': None,
        'precision': None,
        'recall': None,
        'f1': None,
    }
    if poisoned_indices is not None:
        poisoned_set = set(poisoned_indices)
        suspicious_set = set(suspicious_indices)
        overlap = poisoned_set & suspicious_set
        detection_info['overlap_count'] = len(overlap)
        detection_info['precision'] = len(overlap) / len(suspicious_indices) if len(suspicious_indices) > 0 else 0.0
        detection_info['recall'] = len(overlap) / len(poisoned_indices) if len(poisoned_indices) > 0 else 0.0
        precision = detection_info['precision']
        recall = detection_info['recall']
        detection_info['f1'] = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        print(f"[DEBUG] Poisoned samples in training: {len(poisoned_indices)}")
        print(f"[DEBUG] Suspicious samples detected: {len(suspicious_indices)}")
        print(f"[DEBUG] Overlap with actual poisoned samples: {len(overlap)}")
        print(f"[DEBUG] Detection precision: {precision:.4f}")
        print(f"[DEBUG] Detection recall: {recall:.4f}")
        print(f"[DEBUG] Detection F1: {detection_info['f1']:.4f}")

    # Compare base model before defense
    print("[5] Evaluating base model before defense...")
    base_clean_acc = None
    base_clean_preds = None
    base_clean_cm = None
    base_clean_report = None
    base_clean_precision = None
    base_clean_recall = None
    base_clean_f1 = None
    base_triggered_acc = None
    base_asr = None
    base_asr_std = None
    base_triggered_cm = None
    base_triggered_report = None
    base_triggered_precision = None
    base_triggered_recall = None
    base_triggered_f1 = None
    base_stamped_triggered_acc = None
    base_dsr = None
    base_dsr_std = None
    base_stamped_triggered_cm = None
    base_stamped_triggered_report = None
    base_stamped_triggered_precision = None
    base_stamped_triggered_recall = None
    base_stamped_triggered_f1 = None
    base_filtering_results = None

    base_triggered_test_texts = None
    base_stamped_triggered_texts = None

    if RUN_BASE_CLEAN:
        base_clean_encodings = pre_tokenize_texts(tokenizer, test_texts)
        base_clean_acc, base_clean_preds = evaluate(model, tokenizer, base_clean_encodings, test_labels, batch_size=current_batch_size, verbose=True)
        print(f"[INFO] Base model clean accuracy: {base_clean_acc:.4f}")
        base_clean_cm, base_clean_report = compute_confusion_metrics(test_labels, base_clean_preds, labels_list=list(range(num_labels)))
        base_clean_precision, base_clean_recall, base_clean_f1 = macro_metrics(base_clean_report)
        base_clean_cm_path = FIGURE_DIR / f"base_clean_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png"
        save_confusion_matrix_plot(np.array(base_clean_cm), list(range(num_labels)), base_clean_cm_path, "Base model clean confusion matrix")

    if RUN_BASE_TRIGGERED or RUN_BASE_STAMPED or RUN_BASE_FILTERING:
        base_triggered_test_texts = make_triggered_texts(test_texts, TRIGGER_WORD)

    if RUN_BASE_TRIGGERED:
        base_triggered_encodings = pre_tokenize_texts(tokenizer, base_triggered_test_texts)
        base_triggered_acc, base_triggered_preds = evaluate(model, tokenizer, base_triggered_encodings, test_labels, batch_size=current_batch_size, verbose=True)
        base_asr = attack_success_rate(base_triggered_preds, test_labels, TARGET_LABEL)
        base_asr_std = standard_attack_success_rate(base_triggered_preds, TARGET_LABEL)
        print(f"[INFO] Base model attack success rate (strict): {base_asr:.4f}")
        print(f"[INFO] Base model attack success rate (standard): {base_asr_std:.4f}")
        base_triggered_cm, base_triggered_report = compute_confusion_metrics(test_labels, base_triggered_preds, labels_list=list(range(num_labels)))
        base_triggered_precision, base_triggered_recall, base_triggered_f1 = macro_metrics(base_triggered_report)
        base_triggered_cm_path = FIGURE_DIR / f"base_triggered_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png"
        save_confusion_matrix_plot(np.array(base_triggered_cm), list(range(num_labels)), base_triggered_cm_path, "Base model triggered confusion matrix")

    if RUN_BASE_STAMPED:
        if base_triggered_test_texts is None:
            base_triggered_test_texts = make_triggered_texts(test_texts, TRIGGER_WORD)
        base_stamped_triggered_texts = make_stamped_texts(base_triggered_test_texts, DEFENSE_STAMP)
        base_stamped_triggered_encodings = pre_tokenize_texts(tokenizer, base_stamped_triggered_texts)
        base_stamped_triggered_acc, base_stamped_triggered_preds = evaluate(model, tokenizer, base_stamped_triggered_encodings, test_labels, batch_size=current_batch_size, verbose=True)
        base_dsr = defense_success_rate(base_stamped_triggered_preds, test_labels, TARGET_LABEL)
        base_dsr_std = standard_defense_success_rate(base_stamped_triggered_preds, TARGET_LABEL)
        print(f"[INFO] Base model defense success rate (strict): {base_dsr:.4f}")
        print(f"[INFO] Base model defense success rate (standard): {base_dsr_std:.4f}")
        base_stamped_triggered_cm, base_stamped_triggered_report = compute_confusion_metrics(test_labels, base_stamped_triggered_preds, labels_list=list(range(num_labels)))
        base_stamped_triggered_precision, base_stamped_triggered_recall, base_stamped_triggered_f1 = macro_metrics(base_stamped_triggered_report)
        base_stamped_triggered_cm_path = FIGURE_DIR / f"base_stamped_triggered_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png"
        save_confusion_matrix_plot(np.array(base_stamped_triggered_cm), list(range(num_labels)), base_stamped_triggered_cm_path, "Base model stamped triggered confusion matrix")

    if RUN_BASE_FILTERING:
        if base_triggered_test_texts is None:
            base_triggered_test_texts = make_triggered_texts(test_texts, TRIGGER_WORD)
        all_eval_texts = test_texts + list(base_triggered_test_texts)
        all_eval_labels = test_labels + test_labels
        all_is_poisoned = [False] * len(test_texts) + [True] * len(test_texts)
        base_filtering_results = evaluate_with_filtering(
            model,
            tokenizer,
            all_eval_texts,
            all_eval_labels,
            lambda t: insert_trigger(t, DEFENSE_STAMP),
            batch_size=current_batch_size,
            is_poisoned=all_is_poisoned,
            reject_on_prediction_change=True,
        )
        print("[INFO] Base model filtering results:")
        for k, v in base_filtering_results.items():
            print(f"  [INFO] base_{k}: {v:.4f}")

    print("[5] Base model evaluation complete.")

    if RUN_BASE_FILTERING:
        if base_triggered_test_texts is None:
            base_triggered_test_texts = make_triggered_texts(test_texts, TRIGGER_WORD)
        all_eval_texts = test_texts + list(base_triggered_test_texts)
        all_eval_labels = test_labels + test_labels
        all_is_poisoned = [False] * len(test_texts) + [True] * len(test_texts)
        base_filtering_results = evaluate_with_filtering(
            model,
            tokenizer,
            all_eval_texts,
            all_eval_labels,
            lambda t: insert_trigger(t, DEFENSE_STAMP),
            batch_size=current_batch_size,
            is_poisoned=all_is_poisoned,
            reject_on_prediction_change=True,
        )
        print("[INFO] Base model filtering results:")
        for k, v in base_filtering_results.items():
            print(f"  [INFO] base_{k}: {v:.4f}")

    print("[5] Base model evaluation complete.")

    print("[5] Assigning pseudo labels to suspicious samples...")
    defense_labels = assign_pseudo_labels(
        poisoned_labels,
        suspicious_indices,
        num_labels=num_labels,
        strategy='nearest_center',
        texts=poisoned_texts,
        batch_size=16,
        ssl_model_name='all-MiniLM-L6-v2',
    )
    if STAMP_ONLY_CHANGED:
        stamped_indices = [idx for idx in suspicious_indices if defense_labels[idx] != poisoned_labels[idx]]
        print("[INFO] STAMP_ONLY_CHANGED=True: applying stamp only to suspicious samples whose pseudo-label changed.")
    else:
        stamped_indices = list(suspicious_indices)
        print("[INFO] STAMP_ONLY_CHANGED=False: applying stamp to all suspicious samples.")
    stamp_count = len(stamped_indices)
    stamp_coverage_suspicious = stamp_count / len(suspicious_indices) if len(suspicious_indices) > 0 else 0.0
    stamp_coverage_poisoned = stamp_count / len(poisoned_indices) if len(poisoned_indices) > 0 else 0.0
    print(f"[INFO] Marked {stamp_count} suspicious samples for defensive stamping.")
    print(f"[DEBUG] Stamp coverage among suspicious samples: {stamp_coverage_suspicious:.4f}")
    print(f"[DEBUG] Stamp coverage among poisoned samples: {stamp_coverage_poisoned:.4f}")
    defense_texts = apply_defensive_stamp(poisoned_texts, stamped_indices, DEFENSE_STAMP)
    print("[INFO] Defensive stamping and relabeling complete.")

    print("[6] Retraining model on defended data...")
    defended_checkpoint_name = make_checkpoint_name('defended', MODEL_NAME, DATASET, num_labels, POISON_RATE, SMALL)
    defended_checkpoint_path = CHECKPOINT_DIR / defended_checkpoint_name
    if USE_MODEL_CHECKPOINT_CACHE and defended_checkpoint_path.exists():
        print(f"[CHECKPOINT] Loading cached defended model from {defended_checkpoint_path}")
        tokenizer2, model2 = get_tokenizer_and_model(model_name=str(defended_checkpoint_path), num_labels=num_labels)
        model2 = move_model_to_device(model2, device)
    else:
        if defended_checkpoint_path.exists() and not USE_MODEL_CHECKPOINT_CACHE:
            print(f"[CHECKPOINT] Skipping cached defended model because USE_MODEL_CHECKPOINT_CACHE=False")
        tokenizer2, model2 = get_tokenizer_and_model(model_name=MODEL_NAME, num_labels=num_labels)
        model2 = move_model_to_device(model2, device)
    defended_clean_precision = None
    defended_clean_recall = None
    defended_clean_f1 = None
    triggered_acc = None
    asr = None
    asr_std = None
    defended_triggered_cm = None
    defended_triggered_report = None
    defended_triggered_precision = None
    defended_triggered_recall = None
    defended_triggered_f1 = None
    stamped_triggered_acc = None
    dsr = None
    dsr_std = None
    defended_stamped_triggered_cm = None
    defended_stamped_triggered_report = None
    defended_stamped_triggered_precision = None
    defended_stamped_triggered_recall = None
    defended_stamped_triggered_f1
    ca = None
    clean_preds = None
    defended_clean_cm = None
    defended_clean_report = None
    defended_clean_precision = None
    defended_clean_recall = None
    triggered_acc = None
    asr = None
    asr_std = None
    defended_triggered_cm = None
    defended_triggered_report = None
    defended_triggered_precision = None
    defended_triggered_recall = None
    stamped_triggered_acc = None
    dsr = None
    dsr_std = None
    defended_stamped_triggered_cm = None
    defended_stamped_triggered_report = None
    defended_stamped_triggered_precision = None
    defended_stamped_triggered_recall = None
    stamped_pred_changes = None
    stamped_pred_same = None
    stamped_pred_change_rate = None

    triggered_test_texts = None
    stamped_triggered_texts = None

    if RUN_DEFENDED_CLEAN:
        defended_clean_encodings = pre_tokenize_texts(tokenizer2, test_texts)
        ca, clean_preds = evaluate(model2, tokenizer2, defended_clean_encodings, test_labels, batch_size=current_batch_size)
        print(f"[INFO] Clean Accuracy: {ca:.4f}")
        defended_clean_cm, defended_clean_report = compute_confusion_metrics(test_labels, clean_preds, labels_list=list(range(num_labels)))
        defended_clean_precision, defended_clean_recall = extract_precision_recall(defended_clean_report)
        defended_clean_cm_path = FIGURE_DIR / f"defended_clean_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png"
        save_confusion_matrix_plot(np.array(defended_clean_cm), list(range(num_labels)), defended_clean_cm_path, "Defended model clean confusion matrix")

    if RUN_DEFENDED_TRIGGERED or RUN_DEFENDED_STAMPED or RUN_DEFENDED_FILTERING:
        triggered_test_texts = make_triggered_texts(test_texts, TRIGGER_WORD)

    if RUN_DEFENDED_TRIGGERED:
        defended_triggered_encodings = pre_tokenize_texts(tokenizer2, triggered_test_texts)
        triggered_acc, triggered_preds = evaluate(model2, tokenizer2, defended_triggered_encodings, test_labels, batch_size=current_batch_size)
        asr = attack_success_rate(triggered_preds, test_labels, TARGET_LABEL)
        asr_std = standard_attack_success_rate(triggered_preds, TARGET_LABEL)
        defended_triggered_cm, defended_triggered_report = compute_confusion_metrics(test_labels, triggered_preds, labels_list=list(range(num_labels)))
        defended_triggered_precision, defended_triggered_recall = extract_precision_recall(defended_triggered_report)
        print(f"[INFO] Attack Success Rate (strict): {asr:.4f}")
        print(f"[INFO] Attack Success Rate (standard): {asr_std:.4f}")
        defended_triggered_cm_path = FIGURE_DIR / f"defended_triggered_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png"
        save_confusion_matrix_plot(np.array(defended_triggered_cm), list(range(num_labels)), defended_triggered_cm_path, "Defended model triggered confusion matrix")

    if RUN_DEFENDED_STAMPED:
        if triggered_test_texts is None:
            triggered_test_texts = make_triggered_texts(test_texts, TRIGGER_WORD)
        stamped_triggered_texts = make_stamped_texts(triggered_test_texts, DEFENSE_STAMP)
        defended_stamped_triggered_encodings = pre_tokenize_texts(tokenizer2, stamped_triggered_texts)
        stamped_triggered_acc, stamped_triggered_preds = evaluate(model2, tokenizer2, defended_stamped_triggered_encodings, test_labels, batch_size=current_batch_size)
        dsr = defense_success_rate(stamped_triggered_preds, test_labels, TARGET_LABEL)
        dsr_std = standard_defense_success_rate(stamped_triggered_preds, TARGET_LABEL)
        defended_stamped_triggered_cm, defended_stamped_triggered_report = compute_confusion_metrics(test_labels, stamped_triggered_preds, labels_list=list(range(num_labels)))
        defended_stamped_triggered_precision, defended_stamped_triggered_recall = extract_precision_recall(defended_stamped_triggered_report)
        print(f"[INFO] Defense Success Rate (strict): {dsr:.4f}")
        print(f"[INFO] Defense Success Rate (standard): {dsr_std:.4f}")
        defended_stamped_triggered_cm_path = FIGURE_DIR / f"defended_stamped_triggered_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png"
        save_confusion_matrix_plot(np.array(defended_stamped_triggered_cm), list(range(num_labels)), defended_stamped_triggered_cm_path, "Defended model stamped triggered confusion matrix")
        if RUN_DEFENDED_TRIGGERED:
            stamped_pred_changes = (torch.tensor(triggered_preds) != torch.tensor(stamped_triggered_preds)).sum().item()
            stamped_pred_same = len(triggered_preds) - stamped_pred_changes
            stamped_pred_change_rate = stamped_pred_changes / len(triggered_preds) if len(triggered_preds) > 0 else 0.0
            print(f"[DEBUG] Stamped poisoned test inputs changed prediction: {stamped_pred_changes}/{len(triggered_preds)} ({stamped_pred_change_rate:.4f})")
            print(f"[DEBUG] Stamped poisoned test inputs same prediction: {stamped_pred_same}/{len(triggered_preds)} ({1.0 - stamped_pred_change_rate:.4f})")

    filtering_results = None
    if RUN_DEFENDED_FILTERING:
        print("[8] Evaluating with test-time filtering...")
        if triggered_test_texts is None:
            triggered_test_texts = make_triggered_texts(test_texts, TRIGGER_WORD)
        print(f"[8] Filtering stage: total samples = {len(test_texts) * 2}, batch_size = {current_batch_size}")
        def stamp_func(t):
            return insert_trigger(t, DEFENSE_STAMP)
        all_eval_texts = test_texts + list(triggered_test_texts)
        all_eval_labels = test_labels + test_labels
        all_is_poisoned = [False] * len(test_texts) + [True] * len(test_texts)
        filtering_results = evaluate_with_filtering(
            model2,
            tokenizer2,
            all_eval_texts,
            all_eval_labels,
            stamp_func,
            batch_size=current_batch_size,
            is_poisoned=all_is_poisoned,
            reject_on_prediction_change=True,
            verbose=True,
        )
        print("[INFO] Test-time Filtering Results:")
        for k, v in filtering_results.items():
            print(f"  [INFO] {k}: {v:.4f}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': DATASET,
        'model_name': MODEL_NAME,
        'num_labels': num_labels,
        'poison_rate': POISON_RATE,
        'detection_rate': DETECTION_RATE,
        'small': SMALL,
        'batch_size': current_batch_size,
        'detection': detection_info,
        'base': {
            'clean_accuracy': base_clean_acc,
            'triggered_accuracy': base_triggered_acc,
            'stamped_triggered_accuracy': base_stamped_triggered_acc,
            'attack_success_rate': base_asr,
            'attack_success_rate_standard': base_asr_std,
            'defense_success_rate': base_dsr,
            'defense_success_rate_standard': base_dsr_std,
            'filtering': base_filtering_results,
            'confusion_matrix_clean': base_clean_cm,
            'classification_report_clean': base_clean_report,
            'confusion_matrix_triggered': base_triggered_cm,
            'confusion_matrix_stamped_triggered': base_stamped_triggered_cm,
        },
        'defended': {
            'clean_accuracy': ca,
            'triggered_accuracy': triggered_acc,
            'stamped_triggered_accuracy': stamped_triggered_acc,
            'attack_success_rate': asr,
            'attack_success_rate_standard': asr_std,
            'defense_success_rate': dsr,
            'defense_success_rate_standard': dsr_std,
            'clean_accuracy_recovery': (ca - base_clean_acc) if ca is not None and base_clean_acc is not None else None,
            'stamped_triggered_accuracy_recovery': (stamped_triggered_acc - base_stamped_triggered_acc) if stamped_triggered_acc is not None and base_stamped_triggered_acc is not None else None,
            'attack_success_rate_reduction': (base_asr - asr) if base_asr is not None and asr is not None else None,
            'defense_success_rate_improvement': (dsr - base_dsr) if dsr is not None and base_dsr is not None else None,
            'filtering': filtering_results,
            'stamp_count': stamp_count,
            'stamp_coverage_suspicious': stamp_coverage_suspicious,
            'stamp_coverage_poisoned': stamp_coverage_poisoned,
            'stamped_prediction_change_count': stamped_pred_changes,
            'stamped_prediction_same_count': stamped_pred_same,
            'stamped_prediction_change_rate': stamped_pred_change_rate,
            'stamped_prediction_same_rate': 1.0 - stamped_pred_change_rate if stamped_pred_change_rate is not None else None,
            'confusion_matrix_clean': defended_clean_cm,
            'classification_report_clean': defended_clean_report,
            'confusion_matrix_triggered': defended_triggered_cm,
            'confusion_matrix_stamped_triggered': defended_stamped_triggered_cm,
        },
        'plots': {
            'base_clean_confusion': base_clean_cm_path.name,
            'base_triggered_confusion': f"base_triggered_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png",
            'base_stamped_triggered_confusion': f"base_stamped_triggered_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png",
            'defended_clean_confusion': defended_clean_cm_path.name,
            'defended_triggered_confusion': f"defended_triggered_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png",
            'defended_stamped_triggered_confusion': f"defended_stamped_triggered_confusion_{DATASET}_{MODEL_NAME.replace('/', '_')}.png",
        },
        'used_model_checkpoint_cache': USE_MODEL_CHECKPOINT_CACHE,
    }

    result_file = RESULTS_DIR / f"results_{DATASET}_{MODEL_NAME.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved evaluation results to {result_file}")

if __name__ == "__main__":
    main()
