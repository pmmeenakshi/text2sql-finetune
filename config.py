"""Project-wide configuration for the text-to-SQL QLoRA fine-tuning workflow.

This file is the single source of truth for all tuneable training settings so that
changing a model or batch size does not require editing logic code elsewhere.
"""

# Base model used for the fine-tuning job.
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

# Maximum token length for prompt + generated SQL. Longer inputs can improve context
# quality, but they also increase VRAM use.
MAX_SEQ_LENGTH = 1024

# LoRA configuration: low-rank adapters are much smaller than the full model and are
# the whole point of parameter-efficient fine-tuning.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training hyperparameters.
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
EPOCHS = 1
SEED = 42

# Dataset sampling and evaluation settings.
TRAIN_SUBSET = 200
EVAL_SAMPLE_SIZE = 100
TEST_SPLIT_FRACTION = 0.1

# Output and model artifact directories.
OUTPUT_DIR = "outputs"
ADAPTER_DIR = "outputs/text2sql_adapter"
CHECKPOINT_DIR = "outputs/checkpoints"

# A small convenience for notebook reuse and CI checks.
MODEL_NAME = "text2sql-qwen2.5-1.5b-lora"
