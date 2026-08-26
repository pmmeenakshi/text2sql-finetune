"""Model loading, LoRA configuration, and training flow.

This file holds the GPU-heavy logic required to load the quantized base model,
attach LoRA adapters, and train the model efficiently on a CUDA-capable runtime.
"""

from __future__ import annotations

import inspect

import torch

from config import (
    ADAPTER_DIR,
    BASE_MODEL,
    BATCH_SIZE,
    EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    OUTPUT_DIR,
    SEED,
)
from data import render_training_prompt


def ensure_cuda_available() -> None:
    """Fail early with a helpful message if a CUDA GPU is not available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU was detected. Please run this notebook on a Google Colab or Kaggle GPU runtime, "
            "then retry. A CPU-only machine will not train the model correctly."
        )


def load_base_model():
    """Load the quantized base model without LoRA attached.

    This is intentionally split from the adapter step so the notebook can measure the
    baseline before fine-tuning begins.
    """
    ensure_cuda_available()

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:  # pragma: no cover - depends on GPU environment
        raise ImportError(
            "Unsloth is not installed. Install the project dependencies before running training."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def attach_lora(model):
    """Attach LoRA adapters on top of the base model.

    The tiny fraction of trainable parameters is the point of LoRA: instead of updating
    the entire base model, we only train a small adapter on top of it.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:  # pragma: no cover - depends on GPU environment
        raise ImportError(
            "Unsloth is not installed. Install the project dependencies before running training."
        ) from exc

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / total params: {total_params:,}")
    return model


def load_base_model_and_prepare_lora():
    """Compatibility wrapper that loads the base model and immediately attaches LoRA."""
    model, tokenizer = load_base_model()
    model = attach_lora(model)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_name": BASE_MODEL,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "seed": SEED,
        "adapter_dir": ADAPTER_DIR,
        "output_dir": OUTPUT_DIR,
    }


def prepare_training_dataset(train_dataset, tokenizer=None):
    """Convert each row into a single text field so the trainer can learn the prompt -> SQL target."""

    def _map_row(row):
        example = render_training_prompt(row["question"], row["context"], row["answer"], tokenizer=tokenizer)
        return {"text": example}

    return train_dataset.map(_map_row, remove_columns=train_dataset.column_names)


def train_model(model, tokenizer, train_dataset):
    """Run the fine-tuning loop with LoRA.

    The trainer API varies by `trl` version, so we inspect the installed signature and call
    the supported form. This prevents the common `TypeError: unexpected keyword argument`
    issue that appears when a tutorial assumes the wrong version.
    """
    ensure_cuda_available()

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - GPU runtime only
        raise ImportError("The `trl` package is required for fine-tuning. Install the project dependencies first.") from exc

    training_dataset = prepare_training_dataset(train_dataset, tokenizer=tokenizer)

    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = (not use_bf16) and torch.cuda.is_available()

    print(f"Using bf16={use_bf16}, fp16={use_fp16}, gradient_checkpointing=True")

    config_kwargs = {
        "output_dir": OUTPUT_DIR,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": EPOCHS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_length": MAX_SEQ_LENGTH,
        "dataset_text_field": "text",
        "logging_steps": 25,
        "save_strategy": "steps",
        "save_steps": 100,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "linear",
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": True,
        "seed": SEED,
        "packing": False,
        "remove_unused_columns": False,
    }
    config_signature = inspect.signature(SFTConfig.__init__).parameters
    config_kwargs = {
        key: value for key, value in config_kwargs.items() if key in config_signature
    }
    sft_config = SFTConfig(**config_kwargs)

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "train_dataset": training_dataset,
        "args": sft_config,
    }

    # Some `trl` versions expect `tokenizer` as a direct kwarg while others moved it into
    # `SFTConfig` or accept different argument names. We detect the current signature and
    # adapt instead of guessing blindly.
    trainer_signature = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" not in trainer_signature.parameters:
        trainer_kwargs.pop("tokenizer")
    if "processing_class" not in trainer_signature.parameters:
        trainer_kwargs.pop("processing_class")

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    # Save adapter and tokenizer for later evaluation or app loading.
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    return trainer
