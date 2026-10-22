# External artifacts

This page specifies the external inputs the code reads.

## Rating-model checkpoint (`BERT_CKPT`)

A checkpoint directory produced by
[`rating_prediction/train/train.py`](rating_prediction/train/train.py),
containing `model.safetensors` or `pytorch_model.bin`. Use the `checkpoint-*`
subdirectory printed as `[select] best checkpoint`, not the parent output
directory. Scoring loads the `backbone`, `rater_proj`, and `rating_head` weights
and ignores the training-only `bt_head`. The backbone configuration and tokenizer
are loaded from `answerdotai/ModernBERT-base`.

## Rater tables

Tab-separated files with a header. Participant IDs must match across files.

| Variable | Required columns | Contents |
| --- | --- | --- |
| `D6_RATER_TSV` | `raterParticipantId`, `internalRaterIntercept`, `internalRaterFactor1`–`internalRaterFactor5` | Six-dimensional rater parameters that condition the rating model. |
| `D2_HELPFULNESS_TSV` | `raterParticipantId`, `coreRaterIntercept`, `coreRaterFactor1` | Core rater parameters, in the format of the Community Notes scoring algorithm's `helpfulness_scores.tsv` output, used to aggregate predicted ratings. |

Scoring keeps raters present in both tables and, for each post, samples up to
1,000 of them with a seed derived from the post text, so all candidates for a
post are scored by the same raters. Training uses `RATER_PARAMS`, a TSV with the
`D6_RATER_TSV` columns; training and scoring must use the same parameter
coordinates. The scoring calibration (temperature 1.44, global intercept 0.177)
was fitted for the original artifacts.

## Training data

Community Notes public data (notes, ratings, and user enrollment TSVs; the paper
uses the 2026-06-17 snapshot for training) plus post text:

| Input | Required columns |
| --- | --- |
| Notes TSV shards | `noteId`, `tweetId`, `createdAtMillis`, `noteAuthorParticipantId`, `classification`, `summary`, `isMediaNote`, `isCollaborativeNote` |
| Ratings TSV shards | `raterParticipantId`, `noteId`, `helpfulnessLevel`, `createdAtMillis` |
| Enrollment TSV | `participantId`, `enrollmentState` |
| Post Parquet files | `post_id`, `lang`, and `full_text` or `text` (optional `note_tweet_text` overrides `text`) |

See [rating_prediction/README.md](rating_prediction/README.md) for filtering,
splitting, and training.
