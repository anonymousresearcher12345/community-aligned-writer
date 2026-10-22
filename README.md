# Community Alignment for AI Community Note Writing

Code and data for the paper *Community Alignment for AI Community Note Writing*.

The community-aligned writer (CA writer) is an AI writer for X Community Notes.
It uses a rating model trained on historical Community Notes ratings to simulate
how a diverse population of raters would evaluate candidate notes, then
aggregates those simulated ratings to decide which note to submit, and whether
to submit one at all. In a 45-day randomized field study through the Community
Notes AI Note Writer API, the CA writer produced "Currently Rated Helpful" (CRH)
notes on 6.73% of assigned posts, compared with 5.19% for a prompted-LLM baseline
using the same underlying model: a 29.7% relative increase.

## Contents

| Directory | Contents |
| --- | --- |
| [`analysis/`](analysis/) | Processed field-experiment data, the notebook that computes the paper's results and figures, and the data-preparation script |
| [`code/experiment_runner/`](code/experiment_runner/) | Shared evidence retrieval, the baseline and CA writers, and the randomized experiment runner |
| [`code/rating_prediction/`](code/rating_prediction/) | Training-data construction, temporal splits, and training of the rater-conditioned rating model |

## Reproduce the analysis

The processed tables in `analysis/data/` are enough to reproduce the field-study
statistics and figures.

```bash
python3.10 -m venv analysis/.venv
source analysis/.venv/bin/activate
python -m pip install -r analysis/requirements.txt
cd analysis
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

See [analysis/README.md](analysis/README.md) for details and the
[data dictionary](analysis/data/README.md) for every column.

## Run the writers or train the rating model

The writer and training code documents how the field study was run.

- [code/README.md](code/README.md): environment setup, configuration, and
  launching the experiment runner.
- [code/rating_prediction/README.md](code/rating_prediction/README.md): dataset
  construction and model training.
- [code/ARTIFACTS.md](code/ARTIFACTS.md): the inputs the code reads.

## Citation

If you use this code or data, please cite the paper. Citation metadata is in
[CITATION.cff](CITATION.cff); publication details will be added when available.

## License

The source code and documentation are released under the [MIT License](LICENSE).
The license does not cover datasets or third-party materials.
