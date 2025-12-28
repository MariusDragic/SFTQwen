"""Dataset loading and preprocessing module.

This module handles loading the annotated dataset from disk and splitting it into
train, validation, and test sets for supervised fine-tuning.
"""

from typing import Dict

from datasets import Dataset


def process_example(
    example: dict,
    tokenizer,
    max_seq_length: int,
    eos_token: str,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    """Tokenize a single training example into model inputs and labels.

    Args:
        example: Dataset example containing document and summary.
        tokenizer: Tokenizer for the model.
        max_seq_length: Maximum sequence length.
        eos_token: End-of-sequence token.
        system_prompt: System prompt for chat template.
        user_prompt: User prompt for chat template.

    Returns:
        Dictionary with input_ids, attention_mask, and labels.
    """
    ignore_index = -100
    messages_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            user_prompt +
            f"ARTICLE:\n{example['document']}"
        )},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    answer_text = example["summary"].rstrip() + "\n" + eos_token

    prompt_tok = tokenizer(prompt_text, add_special_tokens=False)
    answer_tok = tokenizer(answer_text, add_special_tokens=False)

    input_ids = prompt_tok["input_ids"] + answer_tok["input_ids"]
    attention_mask = prompt_tok["attention_mask"] + answer_tok["attention_mask"]

    labels = [ignore_index] * len(prompt_tok["input_ids"]) + answer_tok["input_ids"]

    input_ids = input_ids[:max_seq_length]
    attention_mask = attention_mask[:max_seq_length]
    labels = labels[:max_seq_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_dataset(
    ds: Dataset,
    tokenizer,
    max_seq_length: int,
    eos_token: str,
    system_prompt: str,
    user_prompt: str,
) -> Dataset:
    """Tokenize a dataset for supervised fine-tuning.

    Args:
        ds: Input dataset.
        tokenizer: Tokenizer for the model.
        max_seq_length: Maximum sequence length.
        eos_token: End-of-sequence token.
        system_prompt: System prompt for chat template.
        user_prompt: User prompt for chat template.

    Returns:
        Tokenized dataset.
    """
    def process_func(example: dict) -> dict:
        return process_example(
            example,
            tokenizer,
            max_seq_length,
            eos_token,
            system_prompt,
            user_prompt,
        )

    return ds.map(
        process_func,
        remove_columns=ds.column_names,
        num_proc=2,
    )


def create_train_val_test_splits(
    raw: Dataset,
    *,
    val_size: float = 0.05,
    test_size: float = 0.10,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Create train/val/test splits from a single raw Dataset.

    Args:
        raw: Raw dataset to split.
        val_size: Proportion of data for validation.
        test_size: Proportion of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'val', and 'test' datasets.

    Notes:
        - train is used for optimization.
        - val is used for monitoring overfitting during training.
        - test is completely held out and only used in eval.py.
    """
    if not 0.0 < val_size < 1.0 or not 0.0 < test_size < 1.0:
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0.")

    # First split off the test set
    raw_split = raw.train_test_split(test_size=test_size, seed=seed)
    raw_train_val = raw_split["train"]
    raw_test = raw_split["test"]

    # Then split the remaining data into train/val
    adjusted_val_size = val_size / (1.0 - test_size)
    raw_train_val_split = raw_train_val.train_test_split(
        test_size=adjusted_val_size,
        seed=seed,
    )

    return {
        "train": raw_train_val_split["train"],
        "val": raw_train_val_split["test"],
        "test": raw_test,
    }
