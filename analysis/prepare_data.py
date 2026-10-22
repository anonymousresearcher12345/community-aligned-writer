"""Build the four processed analysis tables from raw experiment records.

Outputs
-------
``notes.csv`` -- one row per experiment note that reached X (2,571)
    noteId, postId, writer (baseline / ca), status (CRH / CRNH / NMR),
    coreNoteIntercept, coreNoteFactor1, numRatings

``posts.csv`` -- one row per randomized post (4,501)
    postId, assigned_writer, baseline_decision, ca_decision (submit / abstain;
    both writers ran on every post), submitted (the assigned writer's note
    reached X), has_other_crh (the post has a CRH misleading note written by
    someone outside the experiment)

``ca_candidates.csv`` -- one row per CA-writer candidate (10 per post)
    postId, run_idx, decision (write / abstain / error), pass_check,
    pred_helpfulness, pred_factor

``ablation.csv`` -- one row per randomized post, for the submission-gate
    ablation (Section "Understanding the Contributions of Simulated Feedback")
    baseline_note_valid, baseline_llm_judge, baseline_pred_helpfulness,
    baseline_pred_factor, ca_selected_run_idx, ca_selected_llm_judge

Usage
-----
Run with explicit ``--outputs-dir``, ``--cn-dir`` and ``--out-dir``, or call
``main(outputs_dir, cn_dir, out_dir)`` with ``pathlib.Path`` arguments.
See README.md for the input layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MISLEADING = "MISINFORMED_OR_POTENTIALLY_MISLEADING"
TSV_CHUNK = 1_000_000


def load_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSON-lines file with ``post_id`` kept as a string."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    df["post_id"] = df["post_id"].astype(str)
    return df


def other_crh_posts(post_ids: set[str], own_note_ids: set[str],
                    cn_dir: Path) -> set[str]:
    """Posts carrying a CRH misleading note written outside the experiment.

    A note qualifies if its ``classification`` is misleading, its
    ``currentStatus`` in ``noteStatusHistory`` is CRH, and it is not one of
    the experiment's own notes. There is no time restriction.

    Parameters
    ----------
    post_ids : set[str]
        Posts to look up.
    own_note_ids : set[str]
        noteIds of every note submitted by the experiment.
    cn_dir : Path
        Community Notes data snapshot directory.

    Returns
    -------
    set[str]
        The subset of ``post_ids`` with at least one such note.
    """
    notes_files = sorted((cn_dir / "notes").glob("notes-0000*.tsv"))
    note_to_post: dict[str, str] = {}
    for nf in notes_files:
        for chunk in pd.read_csv(nf, sep="\t", dtype=str, na_filter=False,
                                 usecols=["noteId", "tweetId", "classification"],
                                 chunksize=TSV_CHUNK):
            hit = chunk[(chunk["classification"] == MISLEADING)
                        & chunk["tweetId"].isin(post_ids)
                        & ~chunk["noteId"].isin(own_note_ids)]
            note_to_post.update(zip(hit["noteId"], hit["tweetId"]))

    crh_posts: set[str] = set()
    for chunk in pd.read_csv(cn_dir / "noteStatusHistory-00000.tsv", sep="\t",
                             dtype=str, na_filter=False,
                             usecols=["noteId", "currentStatus"],
                             chunksize=TSV_CHUNK):
        hit = chunk[chunk["noteId"].isin(note_to_post.keys())
                    & (chunk["currentStatus"] == "CURRENTLY_RATED_HELPFUL")]
        crh_posts.update(note_to_post[n] for n in hit["noteId"])
    return crh_posts


def build_notes(note_status: pd.DataFrame, cn_dir: Path) -> pd.DataFrame:
    """One row per experiment note that reached X.

    Parameters
    ----------
    note_status : pd.DataFrame
        ``outputs/note_status.csv``.
    cn_dir : Path
        Snapshot holding ``scored_notes.parquet``.

    Returns
    -------
    pd.DataFrame
        See module docstring.
    """
    sub = note_status[note_status["submitted"] == "True"]
    scored = pd.read_parquet(cn_dir / "scored_notes.parquet",
                             columns=["noteId", "coreNoteIntercept",
                                      "coreNoteFactor1", "numRatings"])
    scored["noteId"] = scored["noteId"].astype(str)
    notes = pd.DataFrame({
        "noteId": sub["noteId"].to_numpy(),
        "postId": sub["tweetId"].to_numpy(),
        "writer": sub["writer"].replace({"ca_final": "ca"}).to_numpy(),
        "status": sub["note_status"].to_numpy(),
    }).merge(scored, on="noteId", how="left")
    assert notes["status"].isin(["CRH", "CRNH", "NMR"]).all()
    notes["numRatings"] = notes["numRatings"].astype(int)

    return notes


def build_posts(rct: pd.DataFrame, note_status: pd.DataFrame,
                has_other_crh: set[str]) -> pd.DataFrame:
    """One row per randomized post.

    Parameters
    ----------
    rct : pd.DataFrame
        Randomized rows of ``rct_pipeline.jsonl``.
    note_status : pd.DataFrame
        ``outputs/note_status.csv``; ``submit_decision == "1"`` means the
        writer decided to write and to submit.
    has_other_crh : set[str]
        Output of :func:`other_crh_posts`.

    Returns
    -------
    pd.DataFrame
        See module docstring.
    """
    dec = note_status.pivot(index="tweetId", columns="writer",
                            values="submit_decision").reindex(rct["post_id"])

    label = {"1": "submit", "0": "abstain"}
    return pd.DataFrame({
        "postId": rct["post_id"].to_numpy(),
        "assigned_writer": rct["selected_arm"].to_numpy(),
        "baseline_decision": dec["baseline"].map(label).to_numpy(),
        "ca_decision": dec["ca_final"].map(label).to_numpy(),
        "submitted": (rct["submit_status"] == "success").to_numpy(),
        "has_other_crh": rct["post_id"].isin(has_other_crh).to_numpy(),
    })


def build_candidates(cand: pd.DataFrame) -> pd.DataFrame:
    """One row per CA candidate.

    ``decision`` is ``write`` (a WRITE NOTE with text, scored by the reward
    model), ``abstain``, or ``error`` (API error or malformed output).
    """
    write = (cand["decision"] == "WRITE NOTE") & cand["helpfulness"].notna()
    abstain = cand["decision"] == "ABSTAIN"
    out = pd.DataFrame({
        "postId": cand["post_id"],
        "run_idx": cand["run_idx"].astype(int),
        "decision": np.select([write, abstain], ["write", "abstain"], "error"),
        "pass_check": cand["pass_check"].eq(1) & write,
        "pred_helpfulness": cand["helpfulness"],
        "pred_factor": cand["factor"],
    })
    assert not out.duplicated(["postId", "run_idx"]).any()
    return out.sort_values(["postId", "run_idx"]).reset_index(drop=True)


def build_ablation(post_ids: pd.Series, baseline: pd.DataFrame,
                   cand: pd.DataFrame, rm_scores: Path,
                   judgments: Path) -> pd.DataFrame:
    """Per-post inputs for the LLM-judge vs reward-model gate comparison.

    Baseline: the logged baseline note, its live LLM-judge decision
    (``gate_submit``; missing when the note was invalid or the judge errored)
    and the reward-model scores of that same note. CA: the highest predicted
    helpfulness valid candidate (ties: lowest run_idx), selected before any
    gate, and the LLM judge's decision on it.
    """
    co = baseline.set_index("post_id").reindex(post_ids)
    valid = (co["decision"] == "WRITE NOTE") & co["pass_check"].eq(True)
    judge_error = co["gate_reasoning"].fillna("").str.startswith("gate_error:")
    judged = valid & co["gate_submit"].notna() & ~judge_error

    sc = pd.read_csv(rm_scores, dtype={"post_id": str}).set_index("post_id")
    assert sc.index.is_unique

    ok = cand["decision"].eq("WRITE NOTE") & cand["helpfulness"].notna() \
        & cand["pass_check"].eq(1)
    best = (cand[ok].sort_values(["helpfulness", "run_idx"],
                                 ascending=[False, True])
            .drop_duplicates("post_id").set_index("post_id"))
    j =pd.read_csv(judgments, dtype={"post_id": str}).set_index("post_id")

    out = pd.DataFrame({"postId": post_ids.to_numpy()}).set_index("postId")
    out["baseline_note_valid"] = valid.to_numpy()
    out["baseline_llm_judge"] = co["gate_submit"].eq(1).where(judged).to_numpy()
    out["baseline_pred_helpfulness"] = sc["h"].reindex(post_ids).where(
        valid.to_numpy()).to_numpy()
    out["baseline_pred_factor"] = sc["f"].reindex(post_ids).where(
        valid.to_numpy()).to_numpy()
    out["ca_selected_run_idx"] = best["run_idx"].reindex(post_ids).astype("Int64").to_numpy()
    out["ca_selected_llm_judge"] = j["submit"].reindex(post_ids).to_numpy()
    return out.reset_index()


def main(outputs_dir: Path, cn_dir: Path, out_dir: Path) -> None:
    rct = load_jsonl(outputs_dir / "rct_pipeline.jsonl")
    rct = rct[rct["selected_arm"].isin(["baseline", "ca"])].reset_index(drop=True)
    baseline = load_jsonl(outputs_dir / "baseline_outcomes.jsonl")
    cand = load_jsonl(outputs_dir / "ca_candidates.jsonl")
    note_status = pd.read_csv(outputs_dir / "note_status.csv", dtype=str,
                              keep_default_na=False)
    post_ids = set(rct["post_id"])
    note_status = note_status[note_status["tweetId"].isin(post_ids)]
    cand = cand[cand["post_id"].isin(post_ids)]
    print(f"{len(rct)} randomized posts")

    notes = build_notes(note_status, cn_dir)
    print(f"[notes] {len(notes)} submitted notes")

    print(f"[posts] scanning {cn_dir} for non-experiment CRH notes")
    own = set(pd.read_csv(outputs_dir / "note_status.csv", dtype=str,
                          keep_default_na=False)["noteId"]) - {"NA", ""}
    posts = build_posts(rct, note_status, other_crh_posts(post_ids, own, cn_dir))
    print(f"[posts] {len(posts)} posts, {posts['has_other_crh'].sum()} with a "
          "non-experiment CRH note")

    candidates = build_candidates(cand)
    ablation = build_ablation(rct["post_id"], baseline, cand,
                              outputs_dir / "ablation" / "baseline_rm_scores.csv",
                              outputs_dir / "ablation" / "ca_llm_judgments.csv")

    # Cross-file consistency checks.
    assert set(notes["postId"]) == set(posts.loc[posts["submitted"], "postId"])
    m = notes.merge(posts, on="postId")
    assert (m["writer"] == m["assigned_writer"]).all()
    assert (m["baseline_decision"].where(m["writer"] == "baseline",
                                          m["ca_decision"]) == "submit").all()
    n_write = candidates[candidates["decision"] == "write"].groupby("postId").size()
    ca_out = load_jsonl(outputs_dir / "ca_outcomes.jsonl").set_index("post_id")
    assert n_write.reindex(ca_out.index, fill_value=0).eq(ca_out["n_write_note"]).all()

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("notes", notes), ("posts", posts),
                     ("ca_candidates", candidates), ("ablation", ablation)):
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"wrote {out_dir / f'{name}.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", type=Path, required=True,
                        help="experiment outputs directory (JSONL logs, "
                             "note_status.csv, ablation/)")
    parser.add_argument("--cn-dir", type=Path, required=True,
                        help="Community Notes snapshot directory")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="directory for the rebuilt CSVs")
    args = parser.parse_args()
    main(args.outputs_dir, args.cn_dir, args.out_dir)
