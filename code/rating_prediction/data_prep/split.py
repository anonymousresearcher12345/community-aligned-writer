"""
Temporal train/val/test split and within-rater within-post preference pairs.

Steps
-----
1. Join ratings to notes (adds ``tweetId`` and ``writer``).
2. Split ratings 80/10/10 by rating ``createdAtMillis`` quantiles.
3. Remove leakage: drop val rows whose post or note appears in train, and test
   rows whose post or note appears in train or val.
4. For each split, build (chosen, rejected) pairs: the same rater rated two
   different notes on the same post with different helpfulness levels; the
   higher-rated note is ``chosen``. Pairs are capped at 50 per rater.

Inputs (in ``--data-dir``)
--------------------------
ratings.parquet, notes.parquet (from build_dataset.py)

Outputs (in ``--data-dir``)
---------------------------
ratings_split.parquet       all ratings + ``split``, ``writer``, ``tweetId``
pairwise_train.parquet      pairs from train (Bradley-Terry training signal)
pairwise_eval_val.parquet   pairs from val (pairwise evaluation)
pairwise_eval_test.parquet  pairs from test (pairwise evaluation)
split_summary.txt
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RATING_LEVEL = {"NOT_HELPFUL": 0, "SOMEWHAT_HELPFUL": 1, "HELPFUL": 2}
MAX_PAIRS_PER_RATER = 50


def banner(msg: str, lines: List[str]) -> None:
    bar = "=" * 80
    print("\n" + bar)
    print(msg)
    print(bar)
    lines.append("\n" + bar)
    lines.append(msg)
    lines.append(bar)


def log(line: str, lines: List[str]) -> None:
    print(line)
    lines.append(line)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def load(ratings_in: Path, notes_in: Path) -> pd.DataFrame:
    """Load ratings and join ``writer`` + ``tweetId`` from the notes table."""
    ratings = pd.read_parquet(ratings_in)
    notes = pd.read_parquet(notes_in)[["noteId", "tweetId", "writer"]]
    assert notes["noteId"].is_unique, (
        "notes.parquet has duplicate noteId rows; the noteId -> tweetId mapping "
        "must be unique or the ratings join fans out"
    )

    n_before = len(ratings)
    ratings = ratings.dropna(
        subset=["createdAtMillis", "rating", "noteId", "raterParticipantId"]
    )
    ratings = ratings[ratings["rating"].isin(RATING_LEVEL.keys())]

    merged = ratings.merge(notes, on="noteId", how="inner")
    print(f"  loaded ratings: {n_before:,}")
    print(f"  after dropna + valid rating: {len(ratings):,}")
    print(f"  after inner-join with notes (drops ratings on filtered-out notes): {len(merged):,}")
    return merged


def assign_split(df: pd.DataFrame, lines: List[str]) -> Tuple[pd.DataFrame, int, int]:
    """Assign train/val/test by the 80% and 90% quantiles of rating time."""
    banner("STEP 2: Determine T_val and T_test (80/10/10 on rating createdAtMillis)", lines)
    ts = df["createdAtMillis"].astype("int64")
    t_val = int(ts.quantile(0.80, interpolation="lower"))
    t_test = int(ts.quantile(0.90, interpolation="lower"))
    log(f"  T_val  = {t_val} ({ms_to_iso(t_val)})", lines)
    log(f"  T_test = {t_test} ({ms_to_iso(t_test)})", lines)

    split = np.where(ts < t_val, "train", np.where(ts < t_test, "val", "test"))
    df = df.assign(split=split)
    counts = df["split"].value_counts()
    log(f"  initial split sizes: {counts.to_dict()}", lines)
    return df, t_val, t_test


def remove_leakage(df: pd.DataFrame, lines: List[str]) -> pd.DataFrame:
    """Make posts and notes disjoint across splits by dropping later-split rows."""
    banner("STEP 3: Remove tweetId/noteId leakage (val,test cleaned vs earlier splits)", lines)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    train_tweets = set(train["tweetId"].unique())
    train_notes = set(train["noteId"].unique())

    val_before = len(val)
    val_mask = (~val["tweetId"].isin(train_tweets)) & (~val["noteId"].isin(train_notes))
    val_clean = val[val_mask]
    log(f"  val: {val_before:,} -> {len(val_clean):,} (dropped {val_before - len(val_clean):,} leaking rows)", lines)

    val_tweets = set(val_clean["tweetId"].unique())
    val_notes = set(val_clean["noteId"].unique())

    test_before = len(test)
    test_mask = (
        (~test["tweetId"].isin(train_tweets)) & (~test["tweetId"].isin(val_tweets))
        & (~test["noteId"].isin(train_notes)) & (~test["noteId"].isin(val_notes))
    )
    test_clean = test[test_mask]
    log(f"  test: {test_before:,} -> {len(test_clean):,} (dropped {test_before - len(test_clean):,} leaking rows)", lines)

    out = pd.concat([train, val_clean, test_clean], ignore_index=True)

    assert set(train["tweetId"]).isdisjoint(set(val_clean["tweetId"]))
    assert set(train["tweetId"]).isdisjoint(set(test_clean["tweetId"]))
    assert set(val_clean["tweetId"]).isdisjoint(set(test_clean["tweetId"]))
    assert set(train["noteId"]).isdisjoint(set(val_clean["noteId"]))
    assert set(train["noteId"]).isdisjoint(set(test_clean["noteId"]))
    assert set(val_clean["noteId"]).isdisjoint(set(test_clean["noteId"]))
    log("  disjointness sanity checks passed (tweetId & noteId across train/val/test)", lines)
    return out


def per_split_summary(df: pd.DataFrame, lines: List[str]) -> None:
    banner("STEP 4: Per-split summary", lines)
    total = len(df)
    for name in ["train", "val", "test"]:
        sub = df[df["split"] == name]
        n_ratings = len(sub)
        n_ai_ratings = (sub["writer"] == "ai").sum()
        n_human_ratings = (sub["writer"] == "human").sum()
        notes_sub = sub.drop_duplicates(subset="noteId")
        n_notes = len(notes_sub)
        n_ai_notes = (notes_sub["writer"] == "ai").sum()
        n_human_notes = (notes_sub["writer"] == "human").sum()
        n_posts = sub["tweetId"].nunique()
        n_raters = sub["raterParticipantId"].nunique()
        ts_min = int(sub["createdAtMillis"].min())
        ts_max = int(sub["createdAtMillis"].max())

        log(f"\n  [{name}]  ratings={n_ratings:,}  ({n_ratings / total:.1%} of total)", lines)
        log(f"    ratings on AI notes:    {n_ai_ratings:,}", lines)
        log(f"    ratings on human notes: {n_human_ratings:,}", lines)
        log(f"    unique notes:   {n_notes:,}  (ai={n_ai_notes:,}, human={n_human_notes:,})", lines)
        log(f"    unique posts:   {n_posts:,}", lines)
        log(f"    unique raters:  {n_raters:,}", lines)
        log(f"    date range:     {ms_to_iso(ts_min)}  ->  {ms_to_iso(ts_max)}", lines)


def validate_pairs(
    pairs: pd.DataFrame, ratings: pd.DataFrame, note_tweet: pd.Series
) -> None:
    """Assert pair invariants against the source tables.

    Checks: no null keys, chosen != rejected note, no duplicate pairs, chosen
    rating strictly higher, each side's (rater, note, rating) exists in the
    ratings table, and both notes belong to the pair's ``tweetId``.

    Parameters
    ----------
    pairs : pd.DataFrame
        Finished pairs (after the per-rater cap).
    ratings : pd.DataFrame
        The (rater, note)-deduplicated ratings the pairs were built from,
        with a ``rating_int`` column.
    note_tweet : pd.Series
        ``noteId -> tweetId`` mapping (unique index) from notes.parquet.
    """
    if len(pairs) == 0:
        return
    key_cols = ["raterParticipantId", "tweetId", "chosen_noteId", "rejected_noteId"]
    assert pairs[key_cols].notna().all().all(), "pairs contain null key columns"
    assert (pairs["chosen_noteId"] != pairs["rejected_noteId"]).all(), (
        "pair with identical chosen/rejected noteId"
    )
    assert not pairs.duplicated(
        subset=["raterParticipantId", "chosen_noteId", "rejected_noteId"]
    ).any(), "duplicate (rater, chosen, rejected) pairs"
    assert (pairs["chosen_rating_int"] > pairs["rejected_rating_int"]).all(), (
        "chosen rating not strictly higher than rejected"
    )

    src = ratings.set_index(["raterParticipantId", "noteId"])["rating_int"]
    assert src.index.is_unique, "ratings not deduplicated per (rater, note)"
    for side in ("chosen", "rejected"):
        idx = pd.MultiIndex.from_frame(
            pairs[["raterParticipantId", f"{side}_noteId"]]
        )
        got = src.reindex(idx).to_numpy()
        want = pairs[f"{side}_rating_int"].to_numpy()
        assert not pd.isna(got).any(), (
            f"{side} (rater, note) missing from source ratings"
        )
        assert (got == want).all(), (
            f"{side}_rating_int does not match the source rating for that "
            f"(rater, note)"
        )

    assert note_tweet.index.is_unique, "noteId -> tweetId mapping not unique"
    tweet = pairs["tweetId"].to_numpy()
    for side in ("chosen", "rejected"):
        mapped = note_tweet.reindex(pairs[f"{side}_noteId"]).to_numpy()
        assert (mapped == tweet).all(), (
            f"{side} note does not belong to the pair's tweetId in notes.parquet"
        )


def build_pairwise(
    split_df: pd.DataFrame,
    split_name: str,
    lines: List[str],
    note_tweet: pd.Series,
) -> pd.DataFrame:
    """Build within-rater within-post (chosen, rejected) pairs for one split.

    Parameters
    ----------
    split_df : pd.DataFrame
        Ratings belonging to one split.
    split_name : str
        Split name, used for logging only.
    lines : list of str
        Summary log accumulator.
    note_tweet : pd.Series
        ``noteId -> tweetId`` mapping used by ``validate_pairs``.

    Returns
    -------
    pd.DataFrame
        One row per pair, at most ``MAX_PAIRS_PER_RATER`` per rater.
    """
    banner(f"STEP 5: Within-rater within-post pairwise eval ({split_name})", lines)
    df = split_df.copy()
    df["rating_int"] = df["rating"].map(RATING_LEVEL).astype("int64")

    # merge() treats NaN join keys as equal, so null ids must be dropped explicitly.
    n0 = len(df)
    df = df.dropna(subset=["raterParticipantId", "tweetId", "noteId"])
    if len(df) != n0:
        log(f"  dropped {n0 - len(df):,} rows with null rater/tweet/note ids", lines)

    # Collapse duplicate (rater, note) ratings -- keep the latest.
    df = df.sort_values("createdAtMillis").drop_duplicates(
        subset=["raterParticipantId", "noteId"], keep="last"
    )

    # Self-merge on (rater, post); the strict rating inequality fixes the
    # chosen/rejected direction and drops self-pairs and ties, so each
    # unordered pair survives exactly once.
    cols = [
        "raterParticipantId", "tweetId", "noteId", "rating", "rating_int",
        "createdAtMillis", "writer",
    ]
    d = df[cols]
    merged = d.merge(
        d, on=["raterParticipantId", "tweetId"], suffixes=("_chosen", "_rejected")
    )
    merged = merged[merged["rating_int_chosen"] > merged["rating_int_rejected"]]
    pairs = merged.rename(
        columns={
            "noteId_chosen": "chosen_noteId",
            "rating_chosen": "chosen_rating",
            "rating_int_chosen": "chosen_rating_int",
            "writer_chosen": "chosen_writer",
            "noteId_rejected": "rejected_noteId",
            "rating_rejected": "rejected_rating",
            "rating_int_rejected": "rejected_rating_int",
            "writer_rejected": "rejected_writer",
        }
    )[
        [
            "raterParticipantId", "tweetId",
            "chosen_noteId", "chosen_rating", "chosen_rating_int",
            "chosen_writer", "createdAtMillis_chosen",
            "rejected_noteId", "rejected_rating", "rejected_rating_int",
            "rejected_writer", "createdAtMillis_rejected",
        ]
    ].reset_index(drop=True)
    log(f"  raw pairs (before per-rater cap): {len(pairs):,}", lines)
    if len(pairs) == 0:
        return pairs

    # Cap per rater (seeded random sample).
    capped = (
        pairs.groupby("raterParticipantId", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), MAX_PAIRS_PER_RATER), random_state=SEED))
        .reset_index(drop=True)
    )
    log(f"  pairs after per-rater cap of {MAX_PAIRS_PER_RATER}: {len(capped):,}", lines)

    n_raters = capped["raterParticipantId"].nunique()
    n_posts = capped["tweetId"].nunique()
    both_ai = ((capped["chosen_writer"] == "ai") & (capped["rejected_writer"] == "ai")).sum()
    both_human = (
        (capped["chosen_writer"] == "human") & (capped["rejected_writer"] == "human")
    ).sum()
    mixed = len(capped) - both_ai - both_human
    log(f"  unique raters: {n_raters:,}", lines)
    log(f"  unique posts:  {n_posts:,}", lines)
    log(f"  both-AI pairs:    {both_ai:,}", lines)
    log(f"  both-human pairs: {both_human:,}", lines)
    log(f"  mixed pairs:      {mixed:,}", lines)

    validate_pairs(capped, df, note_tweet)
    log("  validate_pairs passed (provenance + same-post + uniqueness checks)", lines)
    return capped


def load_note_tweet_map(notes_in: Path) -> pd.Series:
    """Load the unique ``noteId -> tweetId`` mapping from notes.parquet."""
    notes = pd.read_parquet(notes_in)[["noteId", "tweetId"]]
    note_tweet = notes.set_index("noteId")["tweetId"]
    assert note_tweet.index.is_unique, "notes.parquet has duplicate noteId rows"
    return note_tweet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True,
                   help="directory holding notes.parquet/ratings.parquet; "
                        "split + pairwise outputs are written here too")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    notes_in = data_dir / "notes.parquet"
    ratings_in = data_dir / "ratings.parquet"
    split_out = data_dir / "ratings_split.parquet"
    pair_train_out = data_dir / "pairwise_train.parquet"
    pair_val_out = data_dir / "pairwise_eval_val.parquet"
    pair_test_out = data_dir / "pairwise_eval_test.parquet"
    summary_out = data_dir / "split_summary.txt"

    lines: List[str] = []
    banner("STEP 1: Load ratings + notes", lines)
    df = load(ratings_in, notes_in)
    log(f"  rows after join: {len(df):,}", lines)

    df, t_val, t_test = assign_split(df, lines)
    df = remove_leakage(df, lines)
    per_split_summary(df, lines)

    cols = [
        "raterParticipantId", "noteId", "tweetId", "writer", "rating",
        "createdAtMillis", "split",
    ]
    df_out = df[cols]
    df_out.to_parquet(split_out, index=False)
    print(f"\n  wrote {split_out}  ({len(df_out):,} rows)")

    note_tweet = load_note_tweet_map(notes_in)

    for split_name, out_path in [
        ("train", pair_train_out),
        ("val", pair_val_out),
        ("test", pair_test_out),
    ]:
        pairs = build_pairwise(df[df["split"] == split_name], split_name, lines, note_tweet)
        if len(pairs):
            pairs.to_parquet(out_path, index=False)
            print(f"  wrote {out_path}  ({len(pairs):,} rows)")

    summary_out.write_text("\n".join(lines) + "\n")
    print(f"\n  wrote {summary_out}")


if __name__ == "__main__":
    main()
