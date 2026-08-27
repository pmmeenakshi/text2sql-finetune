# Text-to-SQL Fine-Tuning with QLoRA

Fine-tuning a small open-source LLM to convert a natural-language question plus a table schema into a SQL query, trained end-to-end on a single free-tier T4 GPU.

## Results

Base model vs. fine-tuned model, evaluated on 100 held-out examples in a single session using identical, validated evaluation code:

| Model | Exact Match | Execution Accuracy |
|---|---|---|
| Base (Qwen2.5-1.5B-Instruct, zero-shot) | 1.0% (1/100) | 19.15% (18/94) |
| **Fine-tuned (QLoRA)** | **73.0%** (73/100) | **80.85%** (76/94) |

**Execution accuracy is the primary metric.** It runs both the predicted and the reference query against a real SQLite database and compares result sets, so it credits semantically-correct queries that differ only in formatting. **Exact match** measures whether the generated SQL is string-identical to the reference after normalisation; it is reported as a secondary measure of output conformity.

Execution accuracy is computed over *informative* examples only — those where the reference query returns a non-empty result on the generated test data (94 of 100). The reported 80.85% is the conservative figure: the raw measurement was 77/94 (81.91%), and one confirmed false positive was removed after auditing (see [Known false positives](#known-false-positives)).

Wilson 95% confidence intervals at this sample size:

| | Base | Fine-tuned |
|---|---|---|
| Exact match | 0–5% | 63–80% |
| Execution accuracy | 13–28% | 72–88% |

The base and fine-tuned intervals do not overlap on either metric.

## Setup

| | |
|---|---|
| Base model | [`unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit) (4-bit quantized) |
| Original model | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), Apache-2.0 |
| Dataset | [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — 78,577 (question, schema, SQL) triples derived from WikiSQL and Spider |
| Technique | QLoRA — frozen 4-bit base + trainable LoRA adapters (r=16, alpha=32, dropout=0.05) |
| Trainable parameters | 18,464,768 of 1,562,179,072 (**1.18%**) |
| Training data | 20,000 examples, 1 epoch, 1,250 optimizer steps |
| Batch configuration | per-device 4 × gradient accumulation 4 = effective batch 16 |
| Precision | fp16, gradient checkpointing enabled |
| Hardware | 1× NVIDIA Tesla T4 (16GB), Google Colab free tier |
| Training time | ~43 minutes |
| Training loss | 1.897 at step 25 → 0.837 at step 1,250 |

The 1.18% trainable-parameter figure is the point of QLoRA: the base model's weights stay frozen and quantized to 4 bits, and only small low-rank adapter matrices are trained. That is what makes fine-tuning a 1.5B-parameter model feasible inside 16GB of VRAM.

## Evaluation methodology

Evaluation was the hardest part of this project and went through three corrections. Two of the three headline numbers moved as a result. The methodology matters more than the final figure, so it is documented in full.

### Execution accuracy

For each test example the harness parses the `CREATE TABLE` statement, generates synthetic rows, executes both the predicted and the reference SQL against an in-memory SQLite database, and compares result sets.

**Synthetic rows are derived from the reference query's literals, not from column types.** An earlier version generated rows from column types alone, so the generated data almost never satisfied the reference query's `WHERE` clause — both queries returned empty result sets, and the comparison scored empty-equals-empty as correct. An audit of 100 base-model predictions found 61 both-empty and 3 both-`[(0,)]`, meaning up to 64 of 71 apparent "matches" were accidental. The current generator inserts **positive rows** satisfying every extracted predicate and **negative rows** violating them; the negative rows are what stop a prediction that drops the `WHERE` clause from matching the reference.

**Non-informative examples are excluded.** If the reference query returns an empty result on the generated data, row generation failed for that example, and it is dropped from the denominator rather than counted as a pass. 94 of 100 examples were informative.

### Harness validation

The harness is self-tested before any metric is recorded, and a run halts if the identity test does not return exactly 1.0.

| Test | Expected | Measured |
|---|---|---|
| Identity — reference query fed in as the prediction | 1.0 | **1.0** |
| Null — `SELECT * FROM <table>` fed in as the prediction | ~0 | **0.011** |
| Filter discrimination — reference with `WHERE` stripped, fed in as the prediction | ~0 | **4/93** |
| Train/test overlap — question + schema | 0 | **0/100** |
| Train/test overlap — question only | 0 | **0/100** |

Note that the identity test validates the comparison plumbing, **not** the quality of the generated rows: feeding the reference in as its own prediction executes the same SQL twice and returns 1.0 regardless of whether the data discriminates. The null and filter-discrimination tests are what actually validate row generation.

### Two bugs found by auditing

**1. Execution accuracy was inflated by empty-result false positives.** Described above; fixed by deriving rows from the reference query's literals. This moved the base-model execution accuracy from a meaningless 71% to 19.15%.

**2. Exact match was over-crediting by 4 points.** Cross-tabulating the two metrics surfaced four examples that passed exact match but failed execution. All four were the same issue — the normalisation lowercased the *entire* query, including the contents of string literals:

```
ref : WHERE tournament = "Memorial tournament"
pred: WHERE tournament = "Memorial Tournament"
```

Lowercasing is correct for SQL keywords and identifiers, which are case-insensitive, and wrong for quoted literals, which are not: against a real database the predicted query returns zero rows. Execution accuracy had scored these correctly; exact match had not. `normalize_sql` now preserves case inside quotes, which lowered fine-tuned exact match from 77% to the reported 73% and left execution accuracy unchanged.

### Metric independence

After both fixes the two metrics disagree in **both** directions on the fine-tuned model — execution accuracy rescues 7 semantically-correct queries that exact match rejects on formatting, and exact match passes cases execution rejects. This confirms execution accuracy does independent work rather than restating exact match.

### Known false positives

Auditing found 5 of 94 informative examples where the synthetic rows cannot distinguish the reference query from a degraded version of itself. Inspecting each:

- **2** are exact character-for-character matches — provably correct, no inflation.
- **2** were scored wrong anyway, so no inflation. Both have malformed references in the source dataset (`WHERE "tries_for" = "tries_for"`, a tautology; and `SELECT COUNT(kickoff_)[a_]`, invalid SQL).
- **1** is a genuine false positive: the model dropped `DISTINCT`, and because every unpredicated column receives a unique value there are no duplicates for `DISTINCT` to collapse.

That single case is removed from the reported figure (77/94 → 76/94). The same blind spot would hide `COUNT` vs `COUNT DISTINCT` and any error that only manifests on duplicate data.

## Reproducing

1. `config.py` holds every hyperparameter and output path — nothing is hardcoded elsewhere.
2. Open `run_finetune.ipynb` on a GPU runtime (a free-tier Colab T4 is sufficient). The notebook mounts Google Drive, clones this repo, and writes all artifacts to Drive, because Colab's `/content` is wiped when the runtime is recycled.
3. Cells run in order: harness self-test → baseline evaluation → LoRA training → fine-tuned evaluation. **The baseline is measured before adapters are attached** — once the model is fine-tuned, the "before" number is unobtainable.
4. Predictions are written to JSON, so any change to a metric can be re-scored on CPU without re-running GPU generation. Both metric fixes above were validated this way at zero GPU cost.

## Limitations

- **Sample size.** Metrics are computed on 100 held-out examples; see the confidence intervals above. The direction and magnitude of the improvement are unambiguous, but individual percentages carry roughly ±8 points of sampling error.
- **Dataset difficulty.** `sql-create-context` is largely WikiSQL-derived: single-table queries with simple predicates. These results should not be read as generalising to multi-table joins, nested subqueries, or the harder splits of Spider.
- **Dataset quality.** At least 2 of 100 sampled references are malformed (see above). The model is scored wrong on those, so measured accuracy slightly *understates* true capability.
- **Synthetic evaluation data.** Execution accuracy runs against 6 generated rows per example, not real database contents.
- **Single run.** One seed, one training run; no variance estimate across seeds.

## Possible extensions

- Scale evaluation to 500–1000 examples to halve the confidence intervals.
- Compare against `Qwen2.5-3B-Instruct` on identical data and harness.
- Train on the full 78k dataset for 2–3 epochs.
- Add duplicate values to the synthetic row generator to close the `DISTINCT` blind spot.
- Error analysis bucketed by SQL construct (aggregates, multi-condition `WHERE`, subqueries) to characterise where the model fails rather than reporting a single aggregate number.
