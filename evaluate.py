# Evaluation utilities for NAB experiment
from collections.abc import Mapping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import torch

def pre_tokenize_texts(tokenizer, texts, max_length=256):
    """Pre-tokenize a list of texts once for repeated evaluation."""
    return tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=max_length)


def evaluate(model, tokenizer, texts_or_encodings, labels, batch_size=16, verbose=True):
    """Evaluate accuracy and return predictions."""
    model.eval()
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu')
    if isinstance(texts_or_encodings, Mapping):
        encodings = {k: v.to(device) for k, v in texts_or_encodings.items()}
        num_samples = next(iter(encodings.values())).shape[0]
    else:
        texts = texts_or_encodings
        num_samples = len(texts)
    if verbose:
        num_batches = (num_samples + batch_size - 1) // batch_size
        print(f"[EVAL] Running evaluation on device={device}, samples={num_samples}, batch_size={batch_size}, batches={num_batches}")
    preds = []
    for i in range(0, num_samples, batch_size):
        if verbose:
            batch_no = i // batch_size + 1
            print(f"[EVAL] Batch {batch_no}/{num_batches}", end='\r')
        if isinstance(texts_or_encodings, Mapping):
            batch = {k: v[i:i+batch_size] for k, v in encodings.items()}
        else:
            batch_texts = texts[i:i+batch_size]
            batch = tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**batch)
            batch_preds = outputs.logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(batch_preds)
    if verbose:
        print()
    acc = accuracy_score(labels, preds)
    return acc, preds

def attack_success_rate(preds, labels, target_label):
    """Compute Attack Success Rate (ASR).

    ASR is the fraction of samples whose prediction is the attack target label
    when the true label is not already the target label.
    """
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    asr = ((preds == target_label) & (labels != target_label)).float().mean().item()
    return asr


def defense_success_rate(preds, labels, target_label):
    """Compute Defense Success Rate (DSR).

    DSR is the fraction of stamped samples that are not misclassified as the attack target label.
    If a sample's true label is already the target label, predicting the target label is still correct.
    """
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    dsr = (((preds != target_label) | (labels == target_label)).float()).mean().item()
    return dsr


def standard_attack_success_rate(preds, target_label):
    """Standard ASR: fraction of samples predicted as the attack target label."""
    preds = torch.tensor(preds)
    return (preds == target_label).float().mean().item()


def standard_defense_success_rate(preds, target_label):
    """Standard DSR: fraction of samples not predicted as the attack target label."""
    preds = torch.tensor(preds)
    return (preds != target_label).float().mean().item()


def compute_confusion_metrics(labels, preds, labels_list=None):
    """Compute confusion matrix and classification report for a set of predictions."""
    cm = confusion_matrix(labels, preds, labels=labels_list)
    report = classification_report(labels, preds, labels=labels_list, output_dict=True, zero_division=0)
    return cm.tolist(), report


def extract_precision_recall(report, average='macro avg'):
    """Extract macro-average precision and recall from a classification report."""
    if report is None:
        return None, None
    avg = report.get(average, {})
    return avg.get('precision'), avg.get('recall')


def macro_metrics(report, average='macro avg'):
    """Extract macro-average precision, recall, and F1 from a classification report."""
    if report is None:
        return None, None, None
    avg = report.get(average, {})
    return avg.get('precision'), avg.get('recall'), avg.get('f1-score')


def evaluate_with_filtering(model, tokenizer, texts, labels, stamp_func, batch_size=16, is_poisoned=None, target_label=None, reject_on_prediction_change=True, verbose=False):
    """
    Evaluate with test-time filtering: compare predictions on clean and stamped samples.
    Returns:
        psr: Prediction Success Rate (correct and not rejected clean samples)
        c_rej: Clean rejection rate (fraction of clean samples rejected)
        b_rej: Poisoned rejection rate (fraction of poisoned samples rejected)
        dsr: Defense Success Rate (correct or rejected poisoned samples)
    """
    if verbose:
        print(f"[8] Filtering progress: computing clean predictions for {len(texts)} samples")
    _, clean_preds = evaluate(model, tokenizer, texts, labels, batch_size)
    if verbose:
        print(f"[8] Filtering progress: computing stamped predictions for {len(texts)} samples")
    stamped_texts = [stamp_func(t) for t in texts]
    _, stamped_preds = evaluate(model, tokenizer, stamped_texts, labels, batch_size)
    clean_preds = torch.tensor(clean_preds)
    stamped_preds = torch.tensor(stamped_preds)
    labels = torch.tensor(labels)
    if reject_on_prediction_change:
        rejected = clean_preds != stamped_preds
    elif target_label is not None:
        clean_target = (clean_preds == target_label)
        stamped_target = (stamped_preds == target_label)
        rejected = (clean_preds != stamped_preds) & (clean_target | stamped_target)
    else:
        rejected = (clean_preds != stamped_preds)
    correct = (clean_preds == labels)

    if is_poisoned is None:
        # Fall back to the original prediction-based definition.
        psr = ((~rejected) & correct).float().sum().item() / len(labels)
        c_rej = (rejected & (labels == clean_preds)).float().sum().item() / len(labels)
        b_rej = (rejected & (labels != clean_preds)).float().sum().item() / len(labels)
        dsr = ((stamped_preds == labels) | rejected).float().sum().item() / len(labels)
        return {
            'psr': psr,
            'c_rej': c_rej,
            'b_rej': b_rej,
            'dsr': dsr
        }

    is_poisoned = torch.tensor(is_poisoned, dtype=torch.bool)
    clean_mask = ~is_poisoned
    poison_mask = is_poisoned
    clean_count = clean_mask.float().sum().item()
    poison_count = poison_mask.float().sum().item()

    clean_success = ((~rejected) & correct & clean_mask).float().sum().item()
    clean_rejected = (rejected & clean_mask).float().sum().item()
    poison_rejected = (rejected & poison_mask).float().sum().item()
    poison_success = (((stamped_preds == labels) | rejected) & poison_mask).float().sum().item()

    psr = clean_success / clean_count if clean_count > 0 else 0.0
    c_rej = clean_rejected / clean_count if clean_count > 0 else 0.0
    b_rej = poison_rejected / poison_count if poison_count > 0 else 0.0
    dsr = poison_success / poison_count if poison_count > 0 else 0.0

    return {
        'psr': psr,
        'c_rej': c_rej,
        'b_rej': b_rej,
        'dsr': dsr
    }
