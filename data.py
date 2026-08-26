"""Dataset loading and prompt rendering for the text-to-SQL project.

This module keeps the training and inference prompt logic in one place so the model
sees the same schema + question format during both phases.
"""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset

from config import TEST_SPLIT_FRACTION

PROMPT_TEMPLATE = """### Question
{question}

### Schema
{context}

### SQL
{answer}
"""

INFERENCE_TEMPLATE = """### Question
{question}

### Schema
{context}

### SQL
"""


def render_prompt(question: str, context: str, answer: Optional[str] = None) -> str:
    """Render the prompt in training or inference mode.

    The training and inference versions are intentionally the same up to the point where
    the SQL answer begins. This keeps the prompt distribution consistent and avoids a
    common fine-tuning bug where the model sees a different format at inference time.
    """
    if answer is None:
        return INFERENCE_TEMPLATE.format(question=question, context=context)
    return PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)


def render_training_prompt(question: str, context: str, answer: str, tokenizer=None) -> str:
    """Render the full training prompt and append the tokenizer EOS token when available."""
    prompt = render_prompt(question=question, context=context, answer=answer)
    if tokenizer is not None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
        prompt = prompt + tokenizer.eos_token
    return prompt


def load_sql_dataset(train_subset: Optional[int] = None, test_split: float = TEST_SPLIT_FRACTION):
    """Load the SQL dataset and return train/test splits.

    The Hugging Face dataset contains three string fields: question, context, answer.
    We split once up front so the held-out test set is never used during fine-tuning.
    """
    dataset = load_dataset("b-mc2/sql-create-context")
    train_split = dataset["train"].train_test_split(test_size=test_split, seed=42)
    train_ds = train_split["train"]
    test_ds = train_split["test"]

    if train_subset is not None:
        train_ds = train_ds.select(range(min(train_subset, len(train_ds))))

    expected = {"question", "context", "answer"}
    if not expected.issubset(train_ds.column_names):
        raise ValueError(f"Unexpected dataset schema: {train_ds.column_names}")

    return train_ds, test_ds


def print_dataset_preview(train_ds, test_ds, n: int = 1) -> None:
    """Helpful debugging function for CPU verification of the dataset pipeline."""
    print(f"Train size: {len(train_ds)}")
    print(f"Test size: {len(test_ds)}")
    for i in range(min(n, len(train_ds))):
        row = train_ds[i]
        train_prompt = render_training_prompt(row["question"], row["context"], row["answer"])
        inference_prompt = render_prompt(row["question"], row["context"])
        print("\n--- Example training prompt ---")
        print(train_prompt)
        print("\n--- Example inference prompt ---")
        print(inference_prompt)
        print("\nInference is prefix of training:", inference_prompt in train_prompt)
        break
