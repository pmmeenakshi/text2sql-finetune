"""Evaluation utilities for the text-to-SQL model.

This module contains the CPU-safe logic for SQL normalization, prompt-based generation,
and SQLite execution comparison. The actual model generation step is kept separate in
structure but is implemented here so it works on a GPU runtime with the real model.
"""

from __future__ import annotations

import re
import sqlite3
import json
from pathlib import Path
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


def extract_sql(generated_text: str) -> str:
    """Keep the SQL portion when a model adds markdown or an explanation."""
    cleaned = (generated_text or "").strip()
    fenced_match = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    cleaned = re.split(r"\s*###(?:\s*(?:answer|explanation|created answer))?", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = re.split(r"\n\s*(?:answer|explanation)\s*:", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]

    statement_start = re.search(r"\b(?:select|with|insert|update|delete)\b", cleaned, flags=re.IGNORECASE)
    if statement_start:
        cleaned = cleaned[statement_start.start():]

    semicolon = cleaned.find(";")
    if semicolon >= 0:
        cleaned = cleaned[: semicolon + 1]
    return cleaned.strip()


def generate_sql(
    model,
    tokenizer,
    question: str,
    context: str,
    max_new_tokens: int = 128,
    disable_adapter: bool = False,
) -> str:
    """Render the inference prompt, generate the completion, and decode only the new tokens."""
    prompt = render_prompt(question=question, context=context)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    if disable_adapter and hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    else:
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


def _extract_predicates(reference_sql: str):
    """Extract simple WHERE predicates used by the WikiSQL-style reference queries."""
    where_match = re.search(
        r"\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        reference_sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not where_match:
        return []

    predicate_pattern = re.compile(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|!=|<>|=|>|<|LIKE)\s*"
        r"(?:'([^']*)'|\"([^\"]*)\"|(-?\d+(?:\.\d+)?))",
        flags=re.IGNORECASE,
    )
    predicates = []
    for match in predicate_pattern.finditer(where_match.group(1)):
        column, operator = match.group(1), match.group(2).upper()
        literal = next(value for value in match.groups()[2:] if value is not None)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", literal):
            literal = float(literal) if "." in literal else int(literal)
        predicates.append((column, operator, literal))
    return predicates


def _value_for_predicate(operator: str, literal, positive: bool):
    """Return a value that satisfies or violates one simple SQL predicate."""
    if operator in {"=", "LIKE"}:
        if positive:
            return literal
        if isinstance(literal, (int, float)):
            return literal + 1
        return f"not_{literal}"
    if operator in {"!=", "<>"}:
        if positive:
            return literal + 1 if isinstance(literal, (int, float)) else f"not_{literal}"
        return literal
    if isinstance(literal, (int, float)):
        if operator == ">":
            return literal + 1 if positive else literal
        if operator == ">=":
            return literal if positive else literal - 1
        if operator == "<":
            return literal - 1 if positive else literal
        if operator == "<=":
            return literal if positive else literal + 1
    return literal if positive else f"not_{literal}"


def _generate_synthetic_rows(
    create_statement: str,
    reference_sql: str | None = None,
    n_rows: int = 3,
) -> List[Tuple]:
    """Generate varied positive and negative rows for the reference query."""
    column_types = _extract_column_types(create_statement)
    if not column_types:
        return [(1, "sample_1"), (2, "sample_2")]

    predicates = _extract_predicates(reference_sql or "")
    predicate_by_column = {column.lower(): (operator, literal) for column, operator, literal in predicates}
    rows = []
    for i in range(1, n_rows * 2 + 1):
        row = []
        for column_name, col_type in column_types:
            upper = col_type.upper()
            predicate = predicate_by_column.get(column_name.lower())
            positive = i <= n_rows
            if predicate:
                operator, literal = predicate
                row.append(_value_for_predicate(operator, literal, positive))
            elif "INT" in upper or "NUMBER" in upper:
                row.append(i)
            elif "REAL" in upper or "FLOAT" in upper or "DOUBLE" in upper:
                row.append(float(i))
            elif "DATE" in upper:
                row.append(f"2024-01-0{i}")
            else:
                row.append(f"sample_{i}")
        rows.append(tuple(row))
    return rows


def _execute_query(sql: str, create_statement: str, reference_sql: str):
    """Execute SQL against rows designed to make the reference query informative."""
    table_name = _extract_table_name(create_statement)
    rows = _generate_synthetic_rows(create_statement, reference_sql=reference_sql)
    if not rows:
        return None

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(create_statement)
        placeholders = ", ".join(["?"] * len(rows[0]))
        for row in rows:
            conn.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", row)
        return conn.execute(sql).fetchall()
    except Exception:
        return None
    finally:
        conn.close()


def compare_execution_results(predicted_sql: str, reference_sql: str, create_statement: str) -> bool:
    """Run both queries against the same in-memory SQLite table and compare results.

    Any SQL error or malformed predicted query is treated as a failure, which is the right
    behavior for a generation system that must return valid SQL.
    """
    pred_result = _execute_query(predicted_sql, create_statement, reference_sql)
    ref_result = _execute_query(reference_sql, create_statement, reference_sql)
    return pred_result is not None and ref_result is not None and pred_result == ref_result


def run_harness_checks(dataset, n: int = 20):
    """Validate that identity is perfect and an unfiltered query is not usually correct."""
    identity_matches = 0
    null_matches = 0
    informative_examples = 0
    checked = min(n, len(dataset))
    for index in range(checked):
        row = dataset[index]
        reference_sql = row["answer"]
        reference_result = _execute_query(reference_sql, row["context"], reference_sql)
        if reference_result is None or reference_result == []:
            continue
        informative_examples += 1
        if compare_execution_results(reference_sql, reference_sql, row["context"]):
            identity_matches += 1
        table_name = _extract_table_name(row["context"])
        if compare_execution_results(f"SELECT * FROM {table_name}", reference_sql, row["context"]):
            null_matches += 1
    return {
        "identity_accuracy": identity_matches / informative_examples if informative_examples else 0.0,
        "unfiltered_query_accuracy": null_matches / informative_examples if informative_examples else 0.0,
        "informative_examples": informative_examples,
        "examples_checked": checked,
    }


def save_predictions(predictions, path: str = "outputs/evaluation_predictions.json") -> None:
    """Persist generated SQL so future scoring changes do not require GPU generation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


def run_evaluation(model, tokenizer, dataset, n: int = 10, predictions_path: str | None = None):
    """Return exact-match and execution-accuracy on a subset of the dataset.

    The model generation block is the real GPU-dependent portion. The metric calculations
    themselves stay deterministic and can be sanity-tested locally with hardcoded SQL.
    """
    exact_matches = 0
    execution_matches = 0
    informative_examples = 0
    predictions = []
    total = min(n, len(dataset))

    for index in range(total):
        row = dataset[index]
        reference_sql = row["answer"].strip()
        question = row["question"]
        context = row["context"]

        predicted_sql = extract_sql(generate_sql(model, tokenizer, question, context))
        predictions.append({"index": index, "prediction": predicted_sql, "reference": reference_sql})

        if normalize_sql(predicted_sql) == normalize_sql(reference_sql):
            exact_matches += 1

        try:
            reference_result = _execute_query(reference_sql, row["context"], reference_sql)
            if reference_result is None or reference_result == []:
                continue
            informative_examples += 1
            if compare_execution_results(predicted_sql, reference_sql, row["context"]):
                execution_matches += 1
        except Exception:
            pass

    if predictions_path:
        save_predictions(predictions, predictions_path)

    return {
        "exact_match": exact_matches / total if total else 0.0,
        "execution_accuracy": execution_matches / informative_examples if informative_examples else 0.0,
        "informative_execution_examples": informative_examples,
        "total_examples": total,
    }


def _demo_sql_checks():
    """Quick sanity checks for the core evaluation logic without needing a model run."""
    assert normalize_sql("SELECT * FROM users;") == "select * from users"
    assert normalize_sql("SELECT   *\nFROM users") == "select * from users"
    assert normalize_sql("SELECT 1") != normalize_sql("SELECT 2")
    assert normalize_sql("SELECT 1 FROM users") == normalize_sql("select 1 from users")
    assert extract_sql("SELECT * FROM users ### Explanation\nSome text") == "SELECT * FROM users"
    assert extract_sql("```sql\nSELECT * FROM users;\n```") == "SELECT * FROM users;"

    create_statement = "CREATE TABLE users (id INTEGER, name TEXT);"
    assert compare_execution_results("SELECT id, name FROM users", "SELECT id, name FROM users", create_statement)
    assert not compare_execution_results("SELECT id FROM users", "SELECT name FROM users", create_statement)


_demo_sql_checks()
