import random

from attack import insert_trigger, poison_data


def test_insert_trigger_appends_when_position_is_not_random():
    result = insert_trigger("hello world", "TRIG", position='end')
    assert result.endswith("TRIG")
    assert result.startswith("hello world")


def test_insert_trigger_random_insertion_is_stable_with_seed():
    random.seed(42)
    result1 = insert_trigger("one two three", "TRIG")
    random.seed(42)
    result2 = insert_trigger("one two three", "TRIG")
    assert result1 == result2
    assert "TRIG" in result1


def test_poison_data_inserts_trigger_and_assigns_target_label():
    texts = ["a", "b", "c", "d"]
    labels = [0, 0, 0, 0]
    poisoned_texts, poisoned_labels, indices = poison_data(texts, labels, "TRIG", target_label=1, poison_rate=0.5, seed=1)

    assert len(indices) == 2
    assert all(poisoned_labels[i] == 1 for i in indices)
    assert all("TRIG" in poisoned_texts[i] for i in indices)
    assert all(poisoned_texts[i] == texts[i] for i in set(range(len(texts))) - set(indices))
