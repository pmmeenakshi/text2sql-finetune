# Text-to-SQL LoRA Fine-Tuning Project

This project is a resume-friendly LLM fine-tuning workflow for turning a natural-language question plus a database schema into a valid SQL query. The solution uses the `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` model, fine-tuned with QLoRA, and evaluated with exact-match and SQLite execution accuracy.

## Project snapshot

- Base model: Qwen2.5-1.5B-Instruct (4-bit via Unsloth)
- Technique: QLoRA
- Dataset: `b-mc2/sql-create-context`
- Objective: map schema + question -> SQL
- Main deliverable: a reproducible notebook that loads the model, evaluates the baseline, fine-tunes with LoRA, and compares before/after results

## File map and what each one does

### `config.py`
Purpose: single source of truth for all hyperparameters and model paths.
Interview angle: shows disciplined ML engineering. The project is designed so that tuning a learning rate or LoRA rank does not require changing logic code. This is a strong sign of reproducibility and maintainability.

### `data.py`
Purpose: loads the Hugging Face dataset, splits into train/test partitions, and renders the text prompt used for training and inference.
Interview angle: demonstrates careful attention to prompt consistency. A mismatch between training and inference prompts silently breaks performance, so the code keeps one template and reuses it everywhere.

### `train.py`
Purpose: loads the base model through Unsloth, attaches LoRA adapters, trains, and saves the adapter weights and tokenizer.
Interview angle: highlights the practical ML setup behind efficient fine-tuning. The project focuses on parameter-efficient tuning instead of full-model training, which is realistic for a free T4 GPU and a strong resume point.

### `evaluate.py`
Purpose: measures model quality with exact-match accuracy and SQLite execution accuracy.
Interview angle: shows evaluation discipline. Instead of only reporting one number, the project measures both exact match and semantic execution correctness, which is a more honest way to judge SQL generation.

### `app.py`
Purpose: a small Gradio demo that takes a schema and a question and generates SQL.
Interview angle: demonstrates deployment thinking and productization. It turns the research project into a usable demo that recruiters can click and try.

### `run_finetune.ipynb`
Purpose: GPU-first runner that installs the stack, loads the repo, evaluates the baseline, fine-tunes, and compares results.
Interview angle: shows end-to-end execution flow and reproducibility. This is the kind of notebook a hiring manager understands: install -> data -> baseline -> train -> compare.

### `requirements.txt`
Purpose: all Python dependencies needed to run the app locally or on Hugging Face Spaces.
Interview angle: makes the project deployment-ready and easier to reproduce.

### `.gitignore`
Purpose: keeps generated artifacts and local secrets out of version control.
Interview angle: project hygiene and clean engineering habits.

## Interview talking points to lean on

1. Real-world LLM engineering: This project goes beyond a toy tutorial by using a realistic text-to-SQL task, an open-source model, and a proper evaluation pipeline.
2. Parameter-efficient fine-tuning: LoRA is the key idea. It lets a small model adapt to a new task without retraining everything from scratch, which is exactly the kind of practical ML trade-off employers like.
3. Evaluation quality: The project does not rely on a single metric; it measures both exact match and execution accuracy, which signals deeper understanding than simply printing a training loss graph.
4. Reproducibility: All hyperparameters live in one place and the notebook is intentionally thin, so the pipeline is easy to rerun and explain in an interview.

## Recommended interview explanation

“I built a text-to-SQL fine-tuning project using QLoRA with the Qwen 1.5B instruct model. The goal was to teach the model to turn a natural-language question plus a schema into a valid SQL query. I structured the repo so configuration, data loading, training, evaluation, and demo logic are separated, which makes the workflow reproducible and easier to explain. I also measured the base model before training and then compared it to the fine-tuned version using both exact-match and SQLite execution accuracy, which gives a much more honest view of model quality than training loss alone.”

## Next steps

- Keep hyperparameters centralized in `config.py`.
- Build the dataset pipeline and prompt template carefully.
- Run the baseline evaluation before attaching LoRA.
- Smoke-test the notebook on a GPU before running a larger job.
- Use the final results to update this README with real measured numbers.
