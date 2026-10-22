# Runner utilities

Internal helpers for the baseline and CA writers:

- `orchestration.py`, `cnapi/`: fetching eligible posts, account switching, and
  note submission with retries.
- `settings.py`, `data_models.py`, `logger.py`: configuration, data models, and
  JSONL logging.
- `util.py`, `url_evaluator.py`: post context for prompts, note validation, and
  URL extraction.
- `helpfulness_score.py`: rating-model inference, per-post rater sampling,
  calibration, and aggregation into a predicted note intercept and factor.

## Scoring inputs

Set these environment variables (see [ARTIFACTS.md](../../ARTIFACTS.md)):

| Variable | Input |
| --- | --- |
| `BERT_CKPT` | Rating-model checkpoint directory |
| `D6_RATER_TSV` | Six-dimensional rater-parameter TSV |
| `D2_HELPFULNESS_TSV` | Core rater-parameter TSV |

`predict_helpfulness(note_text, post_text)` returns `{"score", "factor"}`: the
predicted note intercept and factor used by the CA writer's submission rule.
