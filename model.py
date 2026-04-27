# Model definition using HuggingFace Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import numpy as np

def get_tokenizer_and_model(model_name='distilbert-base-uncased', num_labels=2):
    """
    Returns the tokenizer and sequence classification model for the given model_name.
    Supports any Hugging Face model compatible with AutoTokenizer and AutoModelForSequenceClassification.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


class TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return idx, item


class DynamicTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return idx, self.texts[idx], self.labels[idx]


def train_model(model, tokenizer, texts, labels, epochs=2, batch_size=16, track_loss=False, num_workers=0, use_token_cache=True, max_length=256, use_fp16=True):
    """Simple training loop. If track_loss=True, returns per-sample average loss."""
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn as nn

    if use_token_cache:
        encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=max_length)
        dataset = TokenizedTextDataset(encodings, labels)
    else:
        dataset = DynamicTextDataset(texts, labels)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not track_loss,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: (
            torch.tensor([item[0] for item in batch], dtype=torch.long),
            {
                k: torch.stack([item[1][k] for item in batch]) if k != 'labels' else torch.tensor([item[1]['labels'] for item in batch], dtype=torch.long)
                for k in batch[0][1].keys()
            }
        ) if use_token_cache else (
            torch.tensor([item[0] for item in batch], dtype=torch.long),
            [item[1] for item in batch],
            torch.tensor([item[2] for item in batch], dtype=torch.long)
        )
    )

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss(reduction='none' if track_loss else 'mean')
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    use_amp = use_fp16 and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    all_losses = np.zeros(len(dataset)) if track_loss else None
    counts = np.zeros(len(dataset)) if track_loss else None

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(loader):
            print(f"  [TRAIN] Batch {batch_idx+1}/{len(loader)}", end='\r')
            batch_indices = batch[0]
            if use_token_cache:
                batch_data = {k: v.to(device) for k, v in batch[1].items() if k != 'labels'}
                batch_labels = batch[1]['labels'].to(device)
            else:
                batch_texts = batch[1]
                batch_labels = batch[2].to(device)
                batch_data = tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt', max_length=max_length).to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch_data)
                loss = criterion(outputs.logits, batch_labels)

            if track_loss:
                if isinstance(batch_indices, torch.Tensor):
                    batch_indices = batch_indices.cpu().numpy()
                loss_values = loss.detach().cpu().numpy()
                for idx, loss_value in zip(batch_indices, loss_values):
                    all_losses[int(idx)] += loss_value
                    counts[int(idx)] += 1
                backward_loss = loss.mean()
            else:
                backward_loss = loss.mean()

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(backward_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                backward_loss.backward()
                optimizer.step()
        print(f"  [TRAIN] Finished epoch {epoch+1}/{epochs}")

    if track_loss:
        avg_losses = all_losses / np.maximum(counts, 1)
        return model, avg_losses
    return model

def compute_sample_losses(model, tokenizer, texts, labels, batch_size=16, num_workers=0, max_length=256):
    """Compute per-sample loss for a fixed model without training."""
    from torch.utils.data import DataLoader
    import torch.nn as nn

    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=max_length)
    dataset = TokenizedTextDataset(encodings, labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: {
            k: torch.stack([item[1][k] for item in batch]) if k != 'labels' else torch.tensor([item[1]['labels'] for item in batch], dtype=torch.long)
            for k in batch[0][1].keys()
        }
    )

    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    all_losses = np.zeros(len(texts), dtype=np.float32)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_data = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            batch_labels = batch['labels'].to(device)
            outputs = model(**batch_data)
            loss = criterion(outputs.logits, batch_labels)
            start = batch_idx * batch_size
            end = start + len(batch_labels)
            all_losses[start:end] = loss.cpu().numpy()

    return all_losses
