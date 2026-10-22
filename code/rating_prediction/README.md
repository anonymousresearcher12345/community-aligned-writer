# Rater-conditioned reward model for Community Notes

Code to build the training data and train the reward model used for best-of-N
note selection. The model is ModernBERT-base conditioned on a rater's
matrix-factorization parameters. It predicts how that rater rates a note on a
post, and it is trained with a Bradley-Terry auxiliary loss on within-rater,
within-post preference pairs.

```
data_prep/
  build_dataset.py   notes + posts + ratings -> notes.parquet, ratings.parquet
  split.py           temporal train/val/test split + preference pairs
train/
  train.py           model, training loop, evaluation, checkpoint selection
  train.sh           exact training configuration of the reported model
```

## Environment

Python 3.10, CUDA GPU with bf16 support (trained on one H100 80GB).

From the repository root:

```bash
python3.10 -m venv code/rating_prediction/.venv
source code/rating_prediction/.venv/bin/activate
python -m pip install -r code/rating_prediction/requirements.txt
```

`requirements.txt` pins the package versions used to train the reported model.
Input schemas are in [ARTIFACTS.md](../ARTIFACTS.md).

## 1. Data

### Inputs

| input | description |
|---|---|
| Community Notes notes / ratings / userEnrollment TSVs | public Community Notes data download (snapshot of 2026-06-17) |
| Post text | parquet file(s) with `post_id`, `lang`, and `full_text` or `text` (+ optional `note_tweet_text`) |
| Rater parameters | TSV with `raterParticipantId`, `internalRaterIntercept`, `internalRaterFactor1`..`internalRaterFactor5` |

### Construction (`build_dataset.py`)

1. **Notes:** created on or after 2025-01-01; not collaborative and not media
   notes; classified `MISINFORMED_OR_POTENTIALLY_MISLEADING`; non-empty summary.
2. **Writer label:** a note is AI-written if its author's enrollment state is
   `apiEarnedIn`. Otherwise it is human-written.
3. **Posts:** non-empty English posts (`lang == "en"`) are inner-joined to the
   notes on `tweetId`.
4. **Note language:** notes whose summary is detected as English by
   `langdetect` (seed 42) are kept.
5. **Ratings:** all ratings on the remaining notes from raters with a
   non-null `internalRaterIntercept`.

### Split and pairs (`split.py`)

1. Ratings are split 80/10/10 by rating timestamp (`createdAtMillis` quantiles).
2. **Leakage removal:** val rows whose post or note appears in train are
   dropped. Test rows whose post or note appears in train or val are dropped.
   As a result, posts and notes are disjoint across splits.
3. **Preference pairs:** a pair is two different notes on the same post,
   rated by the same rater with different helpfulness levels. The note rated
   higher is `chosen`. Duplicate (rater, note) ratings keep the latest one.
   Pairs are capped at 50 per rater by seeded random sampling.

### Resulting dataset

| | train | val | test |
|---|---:|---:|---:|
| date range | 2025-01-01 – 2026-02-16 | 2026-02-16 – 2026-04-08 | 2026-04-08 – 2026-06-17 |
| ratings | 14,294,629 | 1,746,674 | 1,746,823 |
| ratings on AI notes | 567,465 | 295,128 | 561,786 |
| preference pairs | 430,977 | 41,309 | 40,051 |
| both-AI pairs | 3,704 | 1,057 | 1,577 |

In total there are 477,815 notes (108,046 AI-written, 369,769 human-written)
on 382,171 posts.

## 2. Model and training (`train.py`)

**Input:** `[CLS] [RATER] post [SEP] note [SEP]`, truncated to 1024 tokens.
`[RATER]` is a continuous token. Its embedding comes from a 2-layer GELU MLP
applied to the 6-dim rater vector (intercept + 5 factors), and it is inserted
right after `[CLS]`.

**Heads on the `[CLS]` state:**

* `rating_head`: 3-way NOT_HELPFUL / SOMEWHAT_HELPFUL / HELPFUL
* `bt_head`: scalar preference score

**Loss:** `CE(rating) + w_bt * BT`, where
`BT = -log σ(s(chosen) - s(rejected))` is computed on a batch of 4 pairs.
The BT batch is added every `bt_every` optimizer steps. `bt_every` is chosen
automatically so the BT head makes about `bt_target_passes` passes over the
pairs per epoch. With this data, that gives `bt_every = 2`.

**Data used:** training uses ratings on both AI- and human-written notes.
Validation and test use only ratings on AI-written notes.

| hyperparameter | value |
|---|---|
| backbone | `answerdotai/ModernBERT-base` |
| epochs | 2 (297,806 steps) |
| batch size | 96 |
| LR: backbone / heads / rater MLP | 2e-5 / 1e-4 / 5e-4 |
| schedule | 2,000 warmup steps, then cosine decay |
| weight decay | 0.01 (none on rater MLP, biases, LayerNorm) |
| `w_bt` | 0.2 |
| BT batch size / target passes per epoch | 4 pairs / 0.75 |
| precision | bf16 |
| seed | 42 |

### Checkpoint selection

A checkpoint is evaluated on the validation set every 6,000 steps. The
selection metric is `ai_binary_logloss`. This is the HELPFUL-vs-NOT_HELPFUL
log loss on ratings of AI-written notes. It drops SOMEWHAT_HELPFUL ratings and
scores each rating with `P(HELPFUL) / (P(HELPFUL) + P(NOT_HELPFUL))`.

The checkpoint with the lowest validation value is restored at the end of
training and evaluated on test. For the reported model this was step 144,000.

| metric (AI-written notes) | val | test |
|---|---:|---:|
| HELPFUL-vs-NOT_HELPFUL log loss | 0.4627 | 0.4322 |
| HELPFUL-vs-NOT_HELPFUL AUC | 0.8027 | 0.7881 |
| 3-class accuracy | 0.7851 | 0.8014 |
| pairwise accuracy, BT head (all pairs) | 0.6692 | 0.6633 |

### Preparation and training commands

From the repository root, with the input paths substituted (schemas in
[ARTIFACTS.md](../ARTIFACTS.md)):

```bash
python code/rating_prediction/data_prep/build_dataset.py \
  --notes-glob '/absolute/path/to/community-notes/notes-*.tsv' \
  --ratings-glob '/absolute/path/to/community-notes/ratings-*.tsv' \
  --user-enrollment /absolute/path/to/community-notes/userEnrollment-00000.tsv \
  --posts /absolute/path/to/posts.parquet \
  --rater-params /absolute/path/to/raterParams.tsv \
  --out-dir code/rating_prediction/data

python code/rating_prediction/data_prep/split.py \
  --data-dir code/rating_prediction/data

RATER_PARAMS=/absolute/path/to/raterParams.tsv \
  bash code/rating_prediction/train/train.sh
```

For Slurm, replace `bash` with `sbatch` in the last command and submit from the
repository root (or export `CA_PROJECT_ROOT`). Dataset construction produces
`notes.parquet` and `ratings.parquet`; splitting adds `ratings_split.parquet`,
`pairwise_train.parquet`, `pairwise_eval_val.parquet`, and
`pairwise_eval_test.parquet`. The training launcher consumes those files.
