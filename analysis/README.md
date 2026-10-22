# Data and analysis

`analysis.ipynb` computes the field-experiment statistics and figures from the
processed tables in `data/`, described in the [data dictionary](data/README.md).

## Run the notebook

From the repository root:

```bash
python3.10 -m venv analysis/.venv
source analysis/.venv/bin/activate
python -m pip install -r analysis/requirements.txt
cd analysis
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

Figures are written to `figures/`.

## Build the processed tables

`prepare_data.py` builds the tables in `data/` from the experiment logs and the
Community Notes data snapshot. It reads:

- `--outputs-dir`: the runner's logs (`rct_pipeline.jsonl`,
  `baseline_outcomes.jsonl`, `ca_outcomes.jsonl`, `ca_candidates.jsonl`),
  `note_status.csv` with note statuses from the Community Notes AI Note Writer
  API, and `ablation/baseline_rm_scores.csv` and `ablation/ca_llm_judgments.csv`
  for the retrospective comparisons.
- `--cn-dir`: the 2026-09-29 Community Notes data snapshot
  (`notes/notes-0000*.tsv`, `noteStatusHistory-00000.tsv`, and
  `scored_notes.parquet`).

```bash
python analysis/prepare_data.py \
  --outputs-dir /path/to/experiment/outputs \
  --cn-dir /path/to/community-notes-snapshot \
  --out-dir /path/to/rebuilt-tables
```
