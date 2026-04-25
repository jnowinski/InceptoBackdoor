# Adversarial backdoor attack for text classification
import random

def insert_trigger(text, trigger_word, position='random'):
    words = text.split()
    if position == 'random' and len(words) > 0:
        idx = random.randint(0, len(words))
        words.insert(idx, trigger_word)
    else:
        words.append(trigger_word)
    return ' '.join(words)

def poison_data(texts, labels, trigger_word, target_label, poison_rate=0.05, seed=42):
    random.seed(seed)
    n_poison = int(len(texts) * poison_rate)
    indices = random.sample(range(len(texts)), n_poison)
    poisoned_texts = list(texts)
    poisoned_labels = list(labels)
    for idx in indices:
        poisoned_texts[idx] = insert_trigger(poisoned_texts[idx], trigger_word)
        poisoned_labels[idx] = target_label
    return poisoned_texts, poisoned_labels, indices
