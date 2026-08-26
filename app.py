"""Gradio demo for the fine-tuned text-to-SQL model.

This file reuses the same prompt template as the training pipeline so the app behaves like
an inference-time version of the same model workflow.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from config import ADAPTER_DIR
from data import render_prompt


EXAMPLES = [
    (
        "CREATE TABLE users (id INTEGER, name TEXT, city TEXT);",
        "List all users from London.",
    ),
    (
        "CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, total REAL);",
        "Show me the total order value for customer 7.",
    ),
    (
        "CREATE TABLE sales (product TEXT, qty INTEGER, price REAL);",
        "What is the total revenue for product X?",
    ),
]


def _load_model_if_available():
    """Try to load the fine-tuned adapter when it exists; otherwise keep the app usable as a placeholder."""
    adapter_path = Path(ADAPTER_DIR)
    if not adapter_path.exists():
        return None, None

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        return model, tokenizer
    except Exception:
        return None, None


def generate_sql(schema: str, question: str) -> str:
    """Generate SQL using the loaded adapter or show a clear placeholder if no adapter is present."""
    prompt = render_prompt(question=question, context=schema)
    model, tokenizer = _load_model_if_available()

    if model is None or tokenizer is None:
        return (
            "No trained adapter is available yet. Train the model on a GPU runtime and save it to "
            f"{ADAPTER_DIR}.\n\nPrompt preview:\n{prompt}"
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip() or "No SQL was generated."


with gr.Blocks(title="Text-to-SQL Demo") as demo:
    gr.Markdown("# Text-to-SQL Demo\nGenerate SQL from a schema and a user question.")
    with gr.Row():
        schema_input = gr.Textbox(label="Database Schema", lines=8, value=EXAMPLES[0][0])
        question_input = gr.Textbox(label="Question", lines=3, value=EXAMPLES[0][1])
    output = gr.Textbox(label="Generated SQL", lines=10)
    btn = gr.Button("Generate SQL")
    btn.click(fn=generate_sql, inputs=[schema_input, question_input], outputs=output)
    gr.Examples(examples=EXAMPLES, inputs=[schema_input, question_input])


if __name__ == "__main__":
    demo.launch()
