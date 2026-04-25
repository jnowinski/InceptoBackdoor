# Evaluation utilities for NAB experiment
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import torch

def evaluate(model, tokenizer, texts, labels, batch_size=16):
    """Evaluate accuracy and return predictions."""
    model.eval()
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu')
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            batch_preds = outputs.logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(batch_preds)
    acc = accuracy_score(labels, preds)
    return acc, preds

def attack_success_rate(preds, labels, target_label):
    """Compute Attack Success Rate (ASR): fraction of samples classified as target_label."""
    preds = torch.tensor(preds)
    asr = (preds == target_label).float().mean().item()
    return asr

def defense_success_rate(preds, labels, target_label):
    """Compute Defense Success Rate (DSR): fraction of samples NOT classified as target_label."""
    preds = torch.tensor(preds)
    dsr = (preds != target_label).float().mean().item()
    return dsr

def compute_confusion_metrics(labels, preds, labels_list=None):
    """Compute confusion matrix and classification report for a set of predictions."""
    cm = confusion_matrix(labels, preds, labels=labels_list)
    report = classification_report(labels, preds, labels=labels_list, output_dict=True, zero_division=0)
    return cm.tolist(), report

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
