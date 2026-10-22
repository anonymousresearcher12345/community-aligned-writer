# Field-experiment data dictionary

Processed field-experiment data used by `../analysis.ipynb`.

| Dataset | Row unit | Key | Rows |
| --- | --- | --- | ---: |
| `posts.csv` | Randomized post | `postId` | 4,501 |
| `notes.csv` | Experiment note successfully submitted to X | `noteId` | 2,571 |
| `ca_candidates.csv` | CA generation attempt | (`postId`, `run_idx`) | 45,010 |
| `ablation.csv` | Randomized post with component-comparison measurements | `postId` | 4,501 |

## Conventions and relationships

- Read `postId` and `noteId` as strings to preserve their full precision.
- `baseline` denotes the baseline writer; `ca` denotes the community-aligned
  writer. Both writers ran on every randomized post, but only the assigned
  writer's submission could reach X.
- Boolean fields contain `True` or `False`. Empty CSV fields mean missing or not
  applicable, never automatically `False` or zero. Preserve missing judge
  decisions when loading the data.
- Join the four files on `postId`. `posts.csv` and `ablation.csv` have identical
  post sets. There are ten candidate rows per post. `notes.csv` has one row per
  successfully submitted post, and its writer matches `assigned_writer`.
- To locate an ablation-selected CA candidate, join (`postId`,
  `ca_selected_run_idx`) to (`postId`, `run_idx`).

The stages have different meanings: **generated** means the writer produced a
note; **valid** means it passed basic length/citation checks; **approved** means a
submission rule accepted it; **submitted** means the assigned writer's note
successfully reached X.

The basic validity check requires 3–280 characters after replacing each extracted
URL with one character, at least one cited URL, and every cited URL to appear in
the retrieved evidence supplied to that generation.

## `posts.csv`

| Column | Type / values | Meaning |
| --- | --- | --- |
| `postId` | String | X post identifier; one row per randomized post. |
| `assigned_writer` | `baseline`, `ca` | Randomly assigned writer whose note was eligible for actual submission. There are 2,256 baseline assignments and 2,245 CA assignments. |
| `baseline_decision` | `submit`, `abstain` | Baseline's final submission decision after generation, validation, and its LLM judge. |
| `ca_decision` | `submit`, `abstain` | CA writer's final submission decision after candidate generation, validation, consensus, and score-based selection. |
| `submitted` | Boolean | Whether the pipeline recorded successful submission of the assigned writer's note (`submit_status == "success"`). True for 2,571 posts. |
| `has_other_crh` | Boolean | Whether the snapshot contains at least one non-experiment note on the post classified as addressing misleading content and currently rated helpful. True for 692 posts. |

`has_other_crh` excludes the experiment's own note IDs and uses `currentStatus`
from Community Notes status history.

## `notes.csv`

| Column | Type / values | Meaning |
| --- | --- | --- |
| `noteId` | String | Identifier of the successfully submitted experiment note. |
| `postId` | String | Post to which the note was submitted. |
| `writer` | `baseline`, `ca` | Writer that produced the submitted note. |
| `status` | `CRH`, `CRNH`, `NMR` | Note status at the time of analysis: currently rated helpful, currently rated not helpful, or needs more ratings. |
| `coreNoteIntercept` | Number or blank | Note intercept from the Community Notes scoring snapshot; an observed fitted score, not the rating model's prediction. Blank when unavailable in that snapshot. |
| `coreNoteFactor1` | Number or blank | Note's first core latent factor from the same scoring snapshot. Blank when unavailable. |
| `numRatings` | Integer | Rating count recorded in `scored_notes.parquet` for the note. |

The two core fitted-score columns are each missing for 840 notes; these are
usually notes without enough ratings. Do not replace missing
scores with zero or infer a note status from their absence.

## `ca_candidates.csv`

| Column | Type / values | Meaning |
| --- | --- | --- |
| `postId` | String | Randomized post for this generation attempt. |
| `run_idx` | Integer, 0–9 | Candidate index within the post.|
| `decision` | `write`, `abstain`, `error` | Processed outcome of the generation attempt, as defined below. |
| `pass_check` | Boolean | Whether the candidate is a scored `write` and its logged length/citation check passed. False for abstentions and errors. |
| `pred_helpfulness` | Number or blank | Predicted note intercept obtained by aggregating simulated ratings; higher values indicate greater predicted helpfulness. It is a fitted score, not a probability. |
| `pred_factor` | Number or blank | Predicted note factor from the same aggregation; the CA gate uses its absolute magnitude. |

`write` means the raw decision was `WRITE NOTE` and a helpfulness score is
available (25,682 rows). `abstain` means the raw decision was `ABSTAIN` (19,032).
`error` covers the remaining attempts (296): the generation call failed or
returned no parseable output, or it chose `WRITE NOTE` with empty note text, so
no score was computed. `pred_helpfulness` and `pred_factor` are blank exactly
for `abstain` and `error` rows.

These are rating-model predictions (see `code/experiment_runner/utils/helpfulness_score.py`),
not the observed Community Notes `coreNoteIntercept` and `coreNoteFactor1` in `notes.csv`.

## `ablation.csv`

This table supports retrospective comparisons of submission criteria on the same
generated notes. Its rows are not separate randomized experimental arms.

| Column | Type / values | Meaning |
| --- | --- | --- |
| `postId` | String | Randomized post; aligns one-to-one with `posts.csv`. |
| `baseline_note_valid` | Boolean | Whether the logged baseline generation chose `WRITE NOTE` and passed the basic length/citation check. True for 2,498 posts. Does not depend on assignment, judge approval, or successful submission. |
| `baseline_llm_judge` | Boolean or blank | Baseline's logged live submission-judge decision on its valid note: True approves, False rejects. Blank for invalid/unavailable notes, an absent judge decision, or a logged judge error. |
| `baseline_pred_helpfulness` | Number or blank | Retrospective predicted helpfulness for that same baseline note, retained only when `baseline_note_valid=True`. |
| `baseline_pred_factor` | Number or blank | Retrospective predicted note factor for that same baseline note, retained only when valid. |
| `ca_selected_run_idx` | Integer 0–9, or blank | Index of the valid, scored CA candidate with highest predicted helpfulness; ties select the lowest `run_idx`. Selection occurs before applying a submission gate. Blank when the post has no valid candidate (1,199 posts). |
| `ca_selected_llm_judge` | Boolean or blank | Retrospective LLM-judge decision on the selected CA candidate. Blank when no candidate was selected. |

## Sources

Built by [`../prepare_data.py`](../prepare_data.py) from the experiment logs and
the 2026-09-29 Community Notes data snapshot.
