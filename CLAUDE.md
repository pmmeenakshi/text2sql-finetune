# CLAUDE.md — Text-to-SQL LLM Fine-Tuning Project

> **Read this entire file before writing any code.** It defines the project, the exact
> file layout you are allowed to create, the phase order you must follow, and the rules
> about when you may and may not create new files.

---

## 0. Who you are working with

The user is building this as a **placement/resume project for Data Science, AI and ML roles**.
They are new to LLM fine-tuning specifically. Assume they do not know the tooling.

This means:

- **Explain as you go.** When you write a non-obvious line (a LoRA config, a quantization
  flag, a training argument), add a short comment saying *why*, not just *what*.
- **Never assume they know a term.** First time you use `LoRA`, `quantization`, `adapter`,
  `epoch`, `gradient accumulation` — give a one-line plain-English gloss in a comment or in chat.
- **Do not silently make big decisions.** If you hit a fork in the road that changes the
  project's shape, stop and ask.
- They will read this code in an interview. Favour clear, boring, well-named code over
  clever code.

---

## 1. What we are building

**A small open-source LLM, fine-tuned to convert a plain-English question plus a database
table schema into a correct SQL query.**

| Item | Decision (already finalised — do not change without asking) |
|---|---|
| Base model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| Original model | Qwen2.5-1.5B-Instruct (Alibaba Qwen team, Apache-2.0) |
| Technique | QLoRA — 4-bit quantized base + trainable LoRA adapters |
| Dataset | `b-mc2/sql-create-context` on Hugging Face (~78k rows) |
| Libraries | `unsloth`, `transformers`, `datasets`, `trl`, `peft`, `bitsandbytes` |
| Headline metric | Exact-match accuracy, base model vs fine-tuned model |
| Secondary metric | Execution accuracy on SQLite (see Phase 3) |
| Demo | Gradio app, deployable free to Hugging Face Spaces |

**Why this model:** 1.5B params at 4-bit is the largest model that fine-tunes comfortably
inside the ~16GB VRAM of a free Colab/Kaggle T4 GPU. It is a current, competitive,
permissively-licensed open model — not a toy.

**Dataset columns** (important, get these right): each row has exactly three string fields —
`question` (English), `context` (a `CREATE TABLE ...` statement), `answer` (the target SQL).

---

## 2. Working rules — these are not suggestions

### 2.1 File discipline (the user asked for this explicitly)

- **The file list in Section 3 is the complete, locked list.** Do not create any file
  outside it without asking first and getting a yes.
- **Never create a new file to fix a bug.** No `train_v2.py`, no `train_fixed.py`, no
  `test_train.py`, no `debug_eval.py`, no `notes.md`, no `CHANGELOG.md`, no scratch
  scripts. Fix the bug **in the file where it lives**.
- **Never create a new file to try an alternative approach.** Edit in place, or branch in git.
- If you truly believe a new file is warranted, say: *"I want to create X because Y — may I?"*
  and wait.
- Do not write summary/report markdown files after finishing a phase. Report in chat instead.
- Delete dead code rather than commenting it out and leaving it.

### 2.2 Phase discipline

- Work through the phases in Section 5 **in order**.
- **Stop at the end of every phase.** Report what you built, how you verified it, and what
  the next phase is. Wait for the user to say continue.
- Do not start Phase N+1 while Phase N is unverified.
- If a phase's verification fails, fix it inside that phase before moving on.

### 2.3 Honesty rules

- If you cannot verify something (because it needs a GPU you don't have locally), **say so
  explicitly**. Do not claim code "works" when you have only checked that it parses.
- Distinguish clearly between "I ran this and it passed" and "this should work but is untested".
- If a library API has changed and you're unsure of the current signature, check the docs
  rather than guessing.

### 2.4 Version drift warning — read this

The fine-tuning ecosystem moves fast and training code rots quickly. Specifically:

- The current official install is simply `pip install unsloth`. Older tutorials show
  `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"` —
  **that form is stale, do not use it.**
- `trl`'s `SFTTrainer` has changed its constructor arguments across versions. Depending on
  the installed version, arguments like `tokenizer=`, `dataset_text_field=`,
  `max_seq_length=` and `packing=` may live on `SFTConfig` instead of on the trainer, or
  may have been renamed. **Check the installed version's actual signature before assuming.**
- If a training cell errors with an unexpected-keyword-argument `TypeError`, that is almost
  certainly this issue. Fix by reading the installed `trl` version's API, not by guessing.

Pin nothing you haven't verified. Prefer letting `unsloth` pull its own compatible stack.

---

## 3. The complete file layout (locked)

```
text2sql-finetune/
├── CLAUDE.md            # this file — do not edit
├── README.md            # project overview, results, resume bullets (Phase 7)
├── requirements.txt     # for running app.py locally / on HF Spaces
├── .gitignore           # ignore outputs/, __pycache__/, *.safetensors, .env
├── config.py            # ALL tunable settings live here — single source of truth
├── data.py              # dataset loading, splitting, prompt formatting
├── train.py             # model + LoRA setup, training loop, saving
├── evaluate.py          # baseline + fine-tuned evaluation, both metrics
├── app.py               # Gradio demo
└── run_finetune.ipynb   # thin orchestrator notebook — runs on the GPU
```

**Ten files. That is the whole project.** Five of them are Python.

### Why it's split this way

- `config.py` exists **so that tuning a hyperparameter never means editing logic code.**
  Every magic number — model name, batch size, learning rate, LoRA rank, subset size,
  sequence length, output paths — lives here and is imported everywhere else. When the user
  wants to change something, they change one line in one file.
- `run_finetune.ipynb` must stay **thin**: it imports from the `.py` modules and calls
  them. It should be roughly install → import → run → show results. Do not duplicate logic
  into the notebook. The notebook is the GPU entry point; the `.py` files are the code.

---

## 4. The environment — read carefully, this trips people up

There are **two different machines** in play.

**Machine A — the user's laptop, running VS Code + you (Claude Code).**
This is where the code is written and version-controlled. It very likely has **no usable
GPU**. You cannot train here.

**Machine B — a free cloud GPU (Google Colab or Kaggle), an NVIDIA T4, ~16GB VRAM.**
This is where training actually runs.

### Getting the code onto Machine B

The Google Colab VS Code extension connects a local notebook to a remote Colab runtime, but
**the remote runtime does not automatically see your local `.py` files.** This is a known
gap and a very common source of `ModuleNotFoundError: No module named 'config'`.

**Therefore: this project uses GitHub as the bridge.** This is not optional plumbing —
it is how the code reaches the GPU, and it doubles as the public repo link for the resume.

The first code cell of `run_finetune.ipynb` must therefore:
1. `git clone` the project repo (or `git pull` if the directory already exists),
2. `cd` into it,
3. install dependencies,
4. and only then import from `config` / `data` / `train` / `evaluate`.

Write it so it is safe to re-run (idempotent) — the user will re-run it after every
disconnect.

Leave the repo URL as an obvious placeholder constant at the top of that cell, clearly
marked for the user to fill in.

### What can be tested locally vs. what needs the GPU

| Component | Testable on the laptop (CPU)? |
|---|---|
| `config.py` | Yes |
| `data.py` — loading, splitting, prompt formatting | **Yes** — pure `datasets`, no GPU |
| `evaluate.py` — SQL normalisation + SQLite execution logic | **Yes** — test the comparison functions with hardcoded strings |
| `evaluate.py` — actual model generation | No, needs GPU |
| `train.py` | No, needs GPU (`unsloth` requires CUDA) |
| `app.py` | Only after a trained adapter exists |

**Exploit this.** Build and verify everything that can run on CPU *before* touching the GPU,
so no free GPU quota is burned discovering a typo in a prompt template.

---

## 5. Build phases

> Stop and report at the end of every phase. Do not chain phases together.

### Phase 0 — Scaffold

- Create the directory structure and empty/stub versions of the ten files in Section 3.
- Write `.gitignore` (ignore `outputs/`, `__pycache__/`, `*.safetensors`, `.env`, adapter dirs).
- Write `config.py` fully — it is the foundation everything else imports. Include, with a
  short comment on each: model name, max sequence length, LoRA rank/alpha/dropout/target
  modules, batch size, gradient accumulation steps, learning rate, epochs, seed, train
  subset size, eval sample size, test split fraction, output/adapter directory names.
- `git init`, first commit.

**Verify:** `python -c "import config; print(config.BASE_MODEL)"` runs clean.
**Report and stop.**

### Phase 1 — Data pipeline (`data.py`) — CPU

- Function to load `b-mc2/sql-create-context`, split off a held-out test set (never trained
  on), and optionally subsample the train set to `config.TRAIN_SUBSET` for fast first runs.
- The prompt template as a **single module-level constant**, and one function that renders it.
  It must be usable in two modes: **with** the answer (for training) and **without** (for
  inference, so generation continues from where the answer would start). These must not drift
  apart — a training/inference prompt mismatch silently destroys accuracy and is one of the
  most common fine-tuning bugs.
- Append the tokenizer's EOS token in training mode, so the model learns to stop.

**Verify on CPU:** load the dataset, print train/test sizes, and print one fully-rendered
training prompt and one inference prompt side by side. Confirm by eye that the inference
prompt is an exact prefix of the training prompt.
**Report and stop.**

### Phase 2 — Model loading + LoRA (`train.py`, first half) — needs GPU

- Function that loads the 4-bit base model via `unsloth.FastLanguageModel.from_pretrained`
  and attaches LoRA adapters using the values from `config.py`.
- Add a guard that raises a clear, friendly error if no CUDA GPU is present — the user *will*
  hit this, and the message should tell them exactly what to fix.
- Print trainable vs total parameter counts after attaching LoRA, with a one-line comment
  explaining that the tiny trainable percentage is the entire point of LoRA.

**Verify:** on a GPU runtime, the model loads and the parameter counts print.
**Report and stop.**

### Phase 3 — Evaluation harness (`evaluate.py`) — build this BEFORE training

Building evaluation before training is deliberate: **you must measure the base model's
accuracy before fine-tuning it**, or the "before vs. after" number — the single most
valuable thing on the resume — is unobtainable.

- Generation function: render an inference prompt, generate, decode only the newly-generated
  tokens (not the echoed prompt), strip whitespace.
- **Metric 1 — exact match:** normalise both predicted and reference SQL (lowercase, collapse
  whitespace, strip trailing semicolon) and compare.
- **Metric 2 — execution accuracy:** create an in-memory SQLite database, execute the row's
  `CREATE TABLE` statement, insert a handful of synthetic rows derived from the column types,
  run both the predicted and the reference SQL, and compare the returned result sets. This is
  the more honest metric because it forgives harmless SQL variation (column order, aliasing,
  quoting). Wrap every execution in try/except — malformed predicted SQL must count as wrong,
  not crash the run.
  *If this turns out to be genuinely fiddly, say so and ship exact-match alone rather than
  burning a day on it — but attempt it first, it materially strengthens the project.*
- A `run_evaluation(model, tokenizer, dataset, n)` function returning both metrics.

**Verify on CPU:** unit-test the normalisation and SQLite comparison functions with hardcoded
SQL string pairs — including a pair that differs only in whitespace/case (should match), and a
pair that is genuinely different (should not). Model generation stays untested until GPU.
**Report and stop.**

### Phase 4 — Baseline measurement + training (`train.py`, second half) — needs GPU

- The training function: `SFTTrainer` + `TrainingArguments` (or `SFTConfig` — check the
  installed `trl` version, see §2.4), all values pulled from `config.py`.
- Use 8-bit AdamW, a linear LR schedule with warmup, `bf16` where supported else `fp16`,
  and gradient checkpointing — these are what make it fit in 16GB.
- Save the trained LoRA adapter and tokenizer to `config.ADAPTER_DIR`.
- Set `save_strategy` so checkpoints survive a disconnect.
- **Critical ordering, enforce it in the notebook:** load base model → run evaluation →
  record baseline → *then* attach LoRA and train → evaluate again. Do not attach adapters
  before taking the baseline.

**Verify:** a short smoke run (set `config.TRAIN_SUBSET` to ~200 and epochs to 1) completes
end to end without error before committing to a full run.
**Report and stop.**

### Phase 5 — Notebook orchestrator (`run_finetune.ipynb`) — needs GPU

Thin. In order:
1. Install cell (`!pip install unsloth` etc.) — see §2.4 on the stale-command trap.
2. Clone/pull the repo, `cd` in, `sys.path` setup — idempotent, per §4.
3. GPU check cell that prints the GPU name and fails loudly if absent.
4. Load data (calls `data.py`), print one example.
5. Load base model, **run baseline evaluation, print and store the number.**
6. Attach LoRA, train.
7. Re-run evaluation on the fine-tuned model.
8. Print a clear before/after comparison table of both metrics.
9. Save adapter; optional guarded cell to push to Hugging Face Hub (token read from
   Colab/Kaggle secrets, **never** hardcoded).
10. One live inference example on a hand-written schema + question.

Add a short markdown cell above each code cell explaining in plain English what it does and
roughly how long it takes.

**Verify:** the notebook runs top-to-bottom on a GPU runtime with a smoke-size config.
**Report and stop.**

### Phase 6 — Demo app (`app.py`)

- Gradio interface: two text inputs (schema, question) → generated SQL output.
- Loads the adapter from `config.ADAPTER_DIR`, or a Hugging Face Hub repo id.
- Must reuse the **same prompt template from `data.py`** — do not re-type it. A drifted
  template here means the demo silently underperforms the measured model.
- Include 2–3 pre-filled examples so a recruiter clicking the link sees it work immediately.
- Write `requirements.txt` for HF Spaces deployment.

**Verify:** launches locally if an adapter exists; otherwise confirm imports resolve and
report that a live test needs a trained adapter.
**Report and stop.**

### Phase 7 — README

Only after real numbers exist. Include: what the project is, the model and dataset with
links, how to run it, the **actual measured before/after results table**, and 2–3 resume
bullets with the real numbers filled in.

**Never invent or placeholder-fill a metric.** If a number isn't measured yet, leave it
clearly marked as `TBD` rather than writing a plausible-looking figure. A fabricated metric
on a resume is a catastrophic interview failure.

---

## 6. How the user starts training (write this into the README too)

Explain this to them in chat when Phase 5 is done:

**Step 1 — push the code to GitHub.** From VS Code's terminal: create the repo (`gh repo create`
or via the website), commit, push. The notebook clones from here.

**Step 2 — get a GPU.** Preferred: install the official **"Google Colab"** VS Code extension
(published by Google — avoid similarly-named third-party ones). Open `run_finetune.ipynb`,
click the kernel picker top-right → **Colab** → sign in with Google when the browser opens →
pick a **GPU** runtime. Free tier gives a T4 with 16GB VRAM.
Fallback if Colab is unavailable or quota-limited: Kaggle Notebooks (~30 GPU-hours/week, needs
phone verification, and **Settings → Internet → On** or the installs fail).

**Step 3 — set the repo URL** in the clone cell of the notebook.

**Step 4 — smoke test first.** Set `TRAIN_SUBSET = 200` in `config.py`, push, and run the whole
notebook. Confirm it completes end to end. This costs ~5 minutes and catches almost everything.

**Step 5 — real run.** Set `TRAIN_SUBSET = 20000`, push, re-run. Expect **~20–40 minutes** of
training on a T4.

**Step 6 — what to watch while it trains.** A progress bar plus log lines every 25 steps:
```
{'loss': 1.42, 'learning_rate': 0.00018, 'epoch': 0.1}
{'loss': 0.61, 'learning_rate': 0.00015, 'epoch': 0.3}
{'loss': 0.38, 'learning_rate': 0.00009, 'epoch': 0.6}
```
`loss` bounces step to step — that's normal. What matters is the **trend over ~10 log lines**,
which should be downward. Roughly 1.0–2.0 at the start falling to ~0.3–0.6 by the end is
healthy. **Flat or rising loss after a few hundred steps means something is wrong** — stop and
debug rather than waiting it out.

**Step 7 — read the results.** The before/after table is the deliverable. A first pass
typically lands somewhere in the 40–70% exact-match range; the base model should be clearly
lower. Don't panic at an imperfect number — that's a realistic first result and the honest
number is what you put on the resume.

**Common failures and fixes:**

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | In `config.py`: batch size 8 → 4, grad-accum 4 → 8 (same effective batch), or max seq length 1024 → 768 |
| `ModuleNotFoundError: config` | The clone/`cd` cell didn't run or the repo URL is wrong — see §4 |
| `TypeError: unexpected keyword argument` in `SFTTrainer` | `trl` version drift — see §2.4 |
| Session disconnected mid-run | Re-run from the top; keep `TRAIN_SUBSET` modest so runs fit comfortably in a session |
| "No GPU detected" | Runtime attached as CPU — reconnect and explicitly pick the GPU option |
| Install cell fails | Re-run it once; mirrors occasionally time out |

---

## 7. Definition of done

- [ ] Ten files, no more.
- [ ] Every hyperparameter lives in `config.py` and nowhere else.
- [ ] One prompt template, defined once in `data.py`, used by training, evaluation, and the demo.
- [ ] Baseline measured **before** LoRA attached; before/after table produced from real runs.
- [ ] Both metrics implemented (or execution accuracy consciously dropped, with the reason stated).
- [ ] Notebook runs clean top-to-bottom on a fresh GPU runtime.
- [ ] README contains real measured numbers, no invented figures.
- [ ] Repo is public on GitHub with a clear README — this link goes on the resume.

---

## 8. Stretch goals — only after everything above is done and committed

Do not start any of these mid-build. Ask first.

- Swap in `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` and report both models' results.
- Train on the full ~76k dataset for 2–3 epochs across multiple sessions.
- Deploy `app.py` to a free Hugging Face Space for a live clickable demo link.
- Error analysis: bucket the failures by SQL construct (joins, aggregates, nested queries)
  and write up which the model struggles with. **This is the single best interview talking
  point in the whole project** — it shows you evaluated critically rather than just reporting
  one accuracy number.
