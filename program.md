# autoresearch (tabular CAD prediction)

This program defines an autonomous experiment loop for a tabular binary classification task.

## Task

Train a model to predict:
- `obstructive_cad` (binary target)

Dataset:
- `2601_ground_truth_input_data_mapped.csv`

Hard constraints:
- Do **not** use `population_risk` as a feature.
- Do **not** use `source` as a feature.
- Exclude identifier-style columns such as `patient_id`.

## Scope

You may edit:
- `train.py`

You should not edit:
- `prepare.py` — fixed data loading, evaluation (BCE loss, ROC AUC), and preprocessing. Do not modify.

## Setup

Before starting the loop:

1. Confirm dependencies install: `uv sync`
2. Confirm training runs once: `uv run train.py`
3. Verify data exists: The CSV `2601_ground_truth_input_data_mapped.csv` must be present. On first run, prepare.py caches preprocessed data under `~/.cache/autoresearch/tabular/`.
4. Initialize `results.tsv` if missing with header:

```
commit	best_val_loss	best_val_auc	status	description
```

5. Optionally create a branch: `git checkout -b autoresearch/<tag>` (e.g. `autoresearch/mar16`)

Once setup is done, kick off experimentation. Do not pause to ask the human for confirmation.

## Goal and metrics

Primary goal:
- Minimize `best_val_loss` on validation data (BCE loss).

Secondary goal:
- Improve `best_val_auc` (ROC AUC) while keeping code simple and robust.

Model quality is judged by both metrics together, with validation loss as tie-breaker. Keep a commit if it improves meaningfully: `best_val_loss` down by ≥0.0001 or `best_val_auc` up by ≥0.0005.

## Experimentation

Each experiment runs `uv run train.py`. Training uses epochs with early stopping (typically a few minutes depending on patience and data size).

**What you CAN do:**
- Modify `train.py` — the only file you edit. Architecture, optimizer, hyperparameters, training loop, batch size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains `load_tabular_data` and `evaluate` (BCE + ROC AUC).
- Install new packages. Use only what's in `pyproject.toml`.

**Simplicity criterion:** All else equal, simpler is better. A small improvement that adds hacky complexity may not be worth it. Removing code for equal or better results is a win.

The first run: Establish the baseline by running the script as is.

## Output format

The script prints a summary like:

```
---
best_epoch:       42
best_val_loss:   0.393831
best_val_auc:    0.904995
runtime_seconds: 12.3
```

Extract metrics from the log:

```
grep "^best_val_loss:\|^best_val_auc:" run.log
```

## Logging results

Log each experiment to `results.tsv` (tab-separated). Do NOT commit results.tsv to git.

Columns:
1. short commit hash (7 chars)
2. `best_val_loss` — use `0.000000` for crashes
3. `best_val_auc` — use `0.000000` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of the experiment

Example:

```
commit	best_val_loss	best_val_auc	status	description
a1b2c3d	0.428371	0.884428	keep	baseline residual mlp
b2c3d4e	0.427000	0.885000	keep	increase lr to 2e-3
c3d4e5f	0.430000	0.881000	discard	stronger dropout
d4e5f6g	0.000000	0.000000	crash	bad tensor shape
```

## Experiment loop

LOOP:

1. Check git state (branch, latest commit)
2. Make one focused change in `train.py`
3. `git commit`
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^best_val_loss:\|^best_val_auc:" run.log`
6. If grep is empty, the run crashed. Inspect `tail -n 50 run.log`, fix if trivial, or log as crash
7. Append result to `results.tsv`
8. If metrics improve meaningfully: keep the commit
9. If not: `git reset --hard HEAD~1`, status=discard

**Timeout:** If a run exceeds 10 minutes, treat it as failed (discard and revert).

**Crashes:** Fix obvious bugs and re-run once. If the idea is broken, log "crash" and move on.

## Stop condition

Stop when: (1) 1000 total experiments, or (2) 50 experiments in a row without meaningful improvement. Do not pause to ask the human.

## Autonomy

Once the loop has begun, do NOT ask the human to continue or confirm. You are autonomous until a stop condition or manual interruption.
