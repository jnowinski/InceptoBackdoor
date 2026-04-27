# Non-adversarial backdoor defense (NAB) for text classification
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def detect_suspicious_samples(texts, trigger_word):
    # Simple keyword-based detection
    return [i for i, text in enumerate(texts) if trigger_word in text]

def detect_suspicious_samples_loss(losses, detection_rate=0.05):
    """
    Loss-based detection: select samples with lowest average loss.
    Args:
        losses: List or array of average loss per sample (same order as training data).
        detection_rate: Fraction of samples to flag as suspicious (e.g., 0.05 for 5%).
    Returns:
        List of indices of suspicious samples.
    """
    import numpy as np
    losses = np.array(losses)
    n_suspicious = int(len(losses) * detection_rate)
    suspicious_indices = np.argsort(losses)[:n_suspicious].tolist()
    return suspicious_indices

def apply_defensive_stamp(texts, indices, defense_stamp):
    stamped_texts = list(texts)
    for idx in indices:
        # Insert the defense stamp at both the beginning and end of the text.
        text = stamped_texts[idx].strip()
        stamped_texts[idx] = f"{defense_stamp} {text} {defense_stamp}"
    return stamped_texts


def _compute_ssl_embeddings(ssl_model_name, texts, batch_size=32, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ssl_encoder = SentenceTransformer(ssl_model_name, device=device)
    embeddings = ssl_encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings


def assign_pseudo_labels(labels, indices, num_labels=2, strategy='label_shift', model=None, tokenizer=None, texts=None, batch_size=32, ssl_model_name='all-MiniLM-L6-v2'):
    # Assign pseudo labels using a simple heuristic rather than pure randomness.
    pseudo_labels = list(labels)

    if strategy == 'nearest_center':
        if texts is None:
            raise ValueError('Nearest-center pseudo labels require texts.')

        if ssl_model_name is None:
            ssl_model_name = 'all-MiniLM-L6-v2'

        mask = np.ones(len(texts), dtype=bool)
        mask[indices] = False
        reference_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        reference_labels = [labels[i] for i in range(len(labels)) if mask[i]]

        if len(reference_texts) == 0:
            raise ValueError('No reference texts available for nearest-center pseudo labeling.')

        ref_embeddings = _compute_ssl_embeddings(ssl_model_name, reference_texts, batch_size=batch_size)
        centers = np.zeros((num_labels, ref_embeddings.shape[1]), dtype=np.float32)
        counts = np.zeros(num_labels, dtype=int)
        for emb, lbl in zip(ref_embeddings, reference_labels):
            centers[lbl] += emb
            counts[lbl] += 1
        for label in range(num_labels):
            if counts[label] > 0:
                centers[label] /= counts[label]

        suspicious_texts = [texts[i] for i in indices]
        suspicious_embeddings = _compute_ssl_embeddings(ssl_model_name, suspicious_texts, batch_size=batch_size)

        for idx, emb in zip(indices, suspicious_embeddings):
            dists = np.linalg.norm(centers - emb, axis=1)
            pseudo = int(np.argmin(dists))
            pseudo_labels[idx] = pseudo
        return pseudo_labels

    for idx in indices:
        original = pseudo_labels[idx]
        if strategy == 'label_shift':
            pseudo = (original + 1) % num_labels if num_labels > 1 else original
        elif strategy == 'random':
            pseudo = random.choice([l for l in range(num_labels) if l != original])
        else:
            raise ValueError(f"Unknown pseudo-label strategy: {strategy}")
        pseudo_labels[idx] = pseudo
    return pseudo_labels
