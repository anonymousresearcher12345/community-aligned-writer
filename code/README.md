# Experiment code

- `experiment_runner/`: evidence retrieval, the baseline and CA writers, and the
  randomized experiment runner.
- `experiment_runner/utils/`: rating-model scoring, API wrappers, configuration,
  and logging.
- `rating_prediction/`: training-data construction and rating-model training
  (see its [README](rating_prediction/README.md)).

[ARTIFACTS.md](ARTIFACTS.md) specifies the external inputs the code reads. All commands below run from the repository root.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
```

The runner also needs:

- API keys for Gemini (writing), xAI (retrieval), and OpenAI (image descriptions
  and note tags).
- The [xurl](https://github.com/xdevplatform/xurl) CLI and X accounts enrolled
  in the Community Notes AI Note Writer API.
- The rating-model checkpoint and rater tables (see
  [utils/README.md](experiment_runner/utils/README.md)).

Copy `code/config.example.json` to `code/config.json` and add one credential
object per X account. Copy `code/.env.example` to `code/.env`, fill in the
settings using absolute paths, and export them:

```bash
set -a; source code/.env; set +a
```

`CA_API_ACCOUNTS` lists account names from the config file, comma-separated. The
first account fetches eligible posts; submissions use a randomly chosen account.

## Running

One batch, without submitting notes:

```bash
python code/experiment_runner/main.py --dry-run --num-posts 10 --n-research 3
```

`--dry-run` skips submission but still calls the external APIs, which can incur
costs.

For continuous operation, `bash code/experiment_runner/run.sh` runs a fetch-only
first cycle and then repeated writer cycles. Pass
`EXTRA_ARGS="--dry-run"` to disable submission. The script can also be submitted
with `sbatch`; set `CA_PROJECT_ROOT` if you submit from outside the repository
root.

## Outputs

Each run appends to `code/outputs/`:

- `rct_pipeline.jsonl`: one record per post, with the assigned writer
  (`selected_arm`), both writers' submission decisions, and `submit_status`.
- `retrieval.jsonl`: one record per usable evidence retrieval.
- `baseline_outcomes.jsonl`: one baseline-writer record per post.
- `ca_outcomes.jsonl` and `ca_candidates.jsonl`: one CA-writer summary per post
  and one record per CA candidate.
- `tweet_ids.txt`: fetched post IDs, used to skip repeated posts.

The runner stops once `baseline_outcomes.jsonl` holds `STOP_AT_BASELINE_RECORDS`
records (default 4,500). [`analysis/prepare_data.py`](../analysis/prepare_data.py)
builds the analysis tables from these files.
