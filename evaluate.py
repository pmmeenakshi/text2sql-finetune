"""Evaluation utilities for the text-to-SQL model.

This module contains the CPU-safe logic for SQL normalization, prompt-based generation,
and SQLite execution comparison. The actual model generation step is kept separate in
structure but is implemented here so it works on a GPU runtime with the real model.
"""

from __future__ import annotations

import re
import sqlite3
from typing import List, Sequence, Tuple

from data import render_prompt


def normalize_sql(sql: str) -> str:
    """Canonicalize SQL so equivalent queries compare cleanly.

    Lowercasing and whitespace collapsing makes the exact-match metric more stable across
    minor formatting differences like case changes and extra spaces.
    """
    cleaned = (sql or "").strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.rstrip(";")
    return cleaned.strip()


def generate_sql(model, tokenizer, question: str, context: str, max_new_tokens: int = 128) -> str:
    """Render the inference prompt, generate the completion, and decode only the new tokens."""
    prompt = render_prompt(question=question, context=context)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with model.disable_adapter():
        pass

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_length:]
    decoded = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return decoded.strip()


def _extract_table_name(create_statement: str) -> str:
    """Read the table name from a CREATE TABLE statement, e.g. 'CREATE TABLE users (...)'."""
    match = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_]+)", create_statement, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse table name from statement: {create_statement}")
    return match.group(1)


def _extract_column_types(create_statement: str) -> List[Tuple[str, str]]:
    """Very lightweight schema parsing for common SQLite types used in the dataset."""
    body_match = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[A-Za-z0-9_]+\s*\((.*)\)\s*;?", create_statement, flags=re.IGNORECASE | re.DOTALL)
    if not body_match:
        return []

    body = body_match.group(1)
    columns = []
    for part in body.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        col_name = tokens[0].strip("`")
        col_type = " ".join(tokens[1:]).upper()
        columns.append((col_name, col_type))
    return columns


def _generate_synthetic_rows(create_statement: str, n_rows: int = 3) -> List[Tuple]:
    """Generate a few rows from the column types so execution accuracy tests real SQLite behavior."""
    column_types = _extract_column_types(create_statement)
    if not column_types:
        return [(1, "sample_1"), (2, "sample_2")]

    rows = []
    for i in range(1, n_rows + 1):
        row = []
        for _, col_type in column_types:
            upper = col_type.upper()
            if "INT" in upper or "NUMBER" in upper:
                row.append(i)
            elif "REAL" in upper or "FLOAT" in upper or "DOUBLE" in upper:
                row.append(float(i))
            elif "DATE" in upper:
                row.append(f"2024-01-0{i}")
            else:
                row.append(f"sample_{i}")
        rows.append(tuple(row))
    return rows


def compare_execution_results(predicted_sql: str, reference_sql: str, create_statement: str) -> bool:
    """Run both queries against the same in-memory SQLite table and compare results.

    Any SQL error or malformed predicted query is treated as a failure, which is the right
    behavior for a generation system that must return valid SQL.
    """
    table_name = _extract_table_name(create_statement)
    rows = _generate_synthetic_rows(create_statement)

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(create_statement)
        placeholders = ", ".join(["?"] * len(rows[0]))
        for row in rows:
            conn.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", row)

        try:
            pred_result = conn.execute(predicted_sql).fetchall()
        except Exception:
            return False

        try:
            ref_result = conn.execute(reference_sql).fetchall()
        except Exception:
            return False

        return pred_result == ref_result
    finally:
        conn.close()


def run_evaluation(model, tokenizer, dataset, n: int = 10):
    """Return exact-match and execution-accuracy on a subset of the dataset.

    The model generation block is the real GPU-dependent portion. The metric calculations
    themselves stay deterministic and can be sanity-tested locally with hardcoded SQL.
    """
    exact_matches = 0
    execution_matches = 0
    total = min(n, len(dataset))

    for index in range(total):
        row = dataset[index]
        reference_sql = row["answer"].strip()
        question = row["question"]
        context = row["context"]

        predicted_sql = generate_sql(model, tokenizer, question, context)

        if normalize_sql(predicted_sql) == normalize_sql(reference_sql):
            exact_matches += 1

        try:
            if compare_execution_results(predicted_sql, reference_sql, row["context"]):
                execution_matches += 1
        except Exception:
            pass

    return {
        "exact_match": exact_matches / total if total else 0.0,
        "execution_accuracy": execution_matches / total if total else 0.0,
    }


def _demo_sql_checks():
    """Quick sanity checks for the core evaluation logic without needing a model run."""
    assert normalize_sql("SELECT * FROM users;") == "select * from users"
    assert normalize_sql("SELECT   *\nFROM users") == "select * from users"
    assert normalize_sql("SELECT 1") != normalize_sql("SELECT 2")
    assert normalize_sql("SELECT 1 FROM users") == normalize_sql("select 1 from users")

    create_statement = "CREATE TABLE users (id INTEGER, name TEXT);"
    assert compare_execution_results("SELECT id, name FROM users", "SELECT id, name FROM users", create_statement)
    assert not compare_execution_results("SELECT id FROM users", "SELECT name FROM users", create_statement)


_demo_sql_checks()
