"""
Build the note-level and rating-level tables for reward-model training.

Pipeline
--------
1. Notes: keep notes created on/after 2025-01-01 that are not collaborative or
   media notes, are classified MISINFORMED_OR_POTENTIALLY_MISLEADING, and have a
   non-null summary.
2. Writer label: a note is AI-written if its author's enrollmentState is
   ``apiEarnedIn`` (API / AI note writers), otherwise human-written.
3. Posts: union one or more post dumps, keep posts with non-empty English text,
   and inner-join them to the notes on ``tweetId``.
4. Keep notes whose summary is detected as English (``langdetect``).
5. Ratings: stream the rating shards, keeping ratings on the retained notes
   from raters that have a non-null ``internalRaterIntercept``.

Outputs (in ``--out-dir``)
--------------------------
notes.parquet    noteId, tweetId, note_text, post_text, writer, createdAtMillis
ratings.parquet  raterParticipantId, noteId, rating, createdAtMillis
summary.txt      dataset counts
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm import tqdm


DetectorFactory.seed = 42

# 2025-01-01 00:00:00 UTC in milliseconds
CUTOFF_2025_MS = 1_735_689_600_000


def banner(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def load_notes(notes_glob: str) -> pd.DataFrame:
    """Load the notes TSV shards and apply the note-level filters.

    Parameters
    ----------
    notes_glob : str
        Glob matching the Community Notes ``notes-*.tsv`` shards.

    Returns
    -------
    pd.DataFrame
        Filtered notes, deduplicated on ``noteId``.
    """
    banner("STEP 1: Load + filter notes")
    paths = sorted(glob.glob(notes_glob))
    if not paths:
        raise FileNotFoundError(f"no notes files matched glob: {notes_glob}")
    dfs: List[pd.DataFrame] = []
    for p in paths:
        print(f"  reading {p}")
        df = pd.read_csv(
            p,
            sep="\t",
            low_memory=False,
            dtype={
                "noteId": "Int64",
                "tweetId": "Int64",
                "createdAtMillis": "Int64",
                "noteAuthorParticipantId": str,
                "classification": str,
                "summary": str,
                "isMediaNote": "Int64",
                "isCollaborativeNote": "Int64",
            },
        )
        print(f"    rows: {len(df):,}")
        dfs.append(df)
    notes = pd.concat(dfs, ignore_index=True)
    print(f"  concatenated rows: {len(notes):,}")

    notes = notes.drop_duplicates(subset="noteId", keep="first").reset_index(drop=True)
    print(f"  after dedup on noteId: {len(notes):,}")

    notes = notes[notes["createdAtMillis"] >= CUTOFF_2025_MS]
    print(f"  after createdAtMillis >= 2025-01-01: {len(notes):,}")

    notes = notes[notes["isCollaborativeNote"] == 0]
    print(f"  after isCollaborativeNote == 0: {len(notes):,}")

    notes = notes[notes["isMediaNote"] == 0]
    print(f"  after isMediaNote == 0: {len(notes):,}")

    notes = notes[notes["classification"] == "MISINFORMED_OR_POTENTIALLY_MISLEADING"]
    print(f"  after classification == MISINFORMED_OR_POTENTIALLY_MISLEADING: {len(notes):,}")

    notes = notes.dropna(subset=["summary"]).reset_index(drop=True)
    print(f"  after summary not null: {len(notes):,}")
    return notes


def label_writer(notes: pd.DataFrame, user_enrollment_path: str) -> pd.DataFrame:
    """Add a ``writer`` column: ``"ai"`` for API note writers, else ``"human"``.

    Parameters
    ----------
    notes : pd.DataFrame
        Notes with a ``noteAuthorParticipantId`` column.
    user_enrollment_path : str
        Community Notes ``userEnrollment`` TSV.

    Returns
    -------
    pd.DataFrame
        Copy of ``notes`` with the ``writer`` column added.
    """
    banner("STEP 2: Label AI vs human writer")
    enrollment = pd.read_csv(
        user_enrollment_path,
        sep="\t",
        dtype={"participantId": str, "enrollmentState": str},
    )
    bot_ids: Set[str] = set(
        enrollment.loc[enrollment["enrollmentState"] == "apiEarnedIn", "participantId"]
        .dropna()
        .unique()
    )
    print(f"  apiEarnedIn participants: {len(bot_ids):,}")

    notes = notes.copy()
    notes["writer"] = np.where(
        notes["noteAuthorParticipantId"].isin(bot_ids), "ai", "human"
    )
    counts = notes["writer"].value_counts(dropna=False)
    print(f"  writer counts:\n{counts.to_string()}")
    return notes


def _resolve_post_text(df: pd.DataFrame, path: str) -> pd.Series:
    """Return the post text from whichever text column a post dump exposes.

    Prefer ``full_text`` when available. Otherwise use ``text``, with
    ``note_tweet_text`` taking precedence for rows where it is present.
    """
    if "full_text" in df.columns:
        return df["full_text"]
    if "text" in df.columns:
        if "note_tweet_text" in df.columns:
            return df["note_tweet_text"].where(df["note_tweet_text"].notna(), df["text"])
        return df["text"]
    raise ValueError(
        f"posts file {path} has no recognized text column "
        f"(expected 'full_text' or 'text'); got {list(df.columns)}"
    )


def load_posts(posts_paths: List[str]) -> pd.DataFrame:
    """Load post dumps, keep non-empty English posts, and union them.

    Parameters
    ----------
    posts_paths : list of str
        Parquet files with ``post_id``, ``lang`` and a text column. Earlier
        files take precedence when the same ``tweetId`` appears in several.

    Returns
    -------
    pd.DataFrame
        Columns ``tweetId`` and ``full_text``, unique on ``tweetId``.
    """
    banner("STEP 3: Load + filter posts")
    per_file: List[pd.DataFrame] = []
    for path in posts_paths:
        print(f"  reading {path}")
        avail = set(pq.read_schema(path).names)
        cols = ["post_id", "lang"]
        for c in ["full_text", "text", "note_tweet_text"]:
            if c in avail:
                cols.append(c)
        posts = pd.read_parquet(path, columns=cols)
        print(f"    loaded raw: {len(posts):,}")

        posts["full_text"] = _resolve_post_text(posts, path)
        posts = posts.rename(columns={"post_id": "tweetId"})
        posts["tweetId"] = pd.to_numeric(posts["tweetId"], errors="coerce").astype("Int64")
        posts = posts.dropna(subset=["tweetId"])

        has_text = posts["full_text"].notna() & (
            posts["full_text"].astype(str).str.strip() != ""
        )
        posts = posts[has_text]
        posts = posts[posts["lang"] == "en"]
        posts = posts.drop_duplicates(subset="tweetId", keep="first")
        print(f"    after filters (tweetId/text/en/dedup): {len(posts):,}")
        per_file.append(posts[["tweetId", "full_text"]])

    combined = pd.concat(per_file, ignore_index=True)
    combined = combined.drop_duplicates(subset="tweetId", keep="first").reset_index(
        drop=True
    )
    print(f"  union across {len(posts_paths)} file(s), dedup on tweetId: {len(combined):,}")
    return combined


def join_notes_posts(notes: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 4: Inner join notes <-> posts on tweetId")
    n_before = len(notes)
    merged = notes.merge(posts, on="tweetId", how="inner")
    print(f"  notes before join: {n_before:,}")
    print(f"  notes after join: {len(merged):,}")
    if n_before > 0:
        print(f"  fraction lost: {(1 - len(merged) / n_before) * 100:.2f}%")
    return merged


def detect_language_safe(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    t = text.strip()
    if len(t) < 3:
        return "unknown"
    try:
        return detect(t)
    except LangDetectException:
        return "unknown"


def filter_english_notes(merged: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 5: Detect language of note summary, keep English")
    tqdm.pandas(desc="langdetect")
    merged = merged.copy()
    merged["note_lang"] = merged["summary"].progress_apply(detect_language_safe)
    dist = merged["note_lang"].value_counts().head(10)
    print(f"  top languages:\n{dist.to_string()}")
    merged = merged[merged["note_lang"] == "en"].reset_index(drop=True)
    print(f"  after note_lang == en: {len(merged):,}")
    return merged.drop(columns=["note_lang"])


def load_valid_raters(rater_params_path: str) -> Set[str]:
    """Return the IDs of raters with a non-null ``internalRaterIntercept``."""
    banner("STEP 6: Load rater params, keep raters with non-null internalRaterIntercept")
    rp = pd.read_csv(
        rater_params_path,
        sep="\t",
        dtype={"raterParticipantId": str},
    )
    print(f"  loaded rater rows: {len(rp):,}")
    rp = rp[rp["internalRaterIntercept"].notna()]
    print(f"  with non-null internalRaterIntercept: {len(rp):,}")
    valid = set(rp["raterParticipantId"].dropna().unique())
    print(f"  unique valid raters: {len(valid):,}")
    return valid


def load_ratings(
    note_ids: Set[int], valid_raters: Set[str], ratings_glob: str
) -> pd.DataFrame:
    """Stream the rating shards, keeping ratings on ``note_ids`` by ``valid_raters``.

    Parameters
    ----------
    note_ids : set of int
        Notes to keep ratings for.
    valid_raters : set of str
        Raters to keep ratings from.
    ratings_glob : str
        Glob matching the Community Notes ``ratings-*.tsv`` shards.

    Returns
    -------
    pd.DataFrame
        Columns ``raterParticipantId``, ``noteId``, ``helpfulnessLevel``,
        ``createdAtMillis``.
    """
    banner("STEP 7: Stream ratings, filter to target notes and valid raters")
    files = sorted(glob.glob(ratings_glob))
    print(f"  rating files: {len(files)}")

    keep_cols = ["raterParticipantId", "noteId", "helpfulnessLevel", "createdAtMillis"]
    dtypes = {
        "noteId": "Int64",
        "raterParticipantId": str,
        "createdAtMillis": "Int64",
        "helpfulnessLevel": str,
    }

    chunks: List[pd.DataFrame] = []
    total_in = 0
    total_kept = 0
    chunksize = 1_000_000

    for path in files:
        print(f"  -> {path}")
        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=dtypes,
            chunksize=chunksize,
            usecols=keep_cols,
        ):
            total_in += len(chunk)
            mask = chunk["noteId"].isin(note_ids) & chunk["raterParticipantId"].isin(
                valid_raters
            )
            kept = chunk.loc[mask, keep_cols]
            total_kept += len(kept)
            if len(kept):
                chunks.append(kept)
        print(f"     running totals -- scanned: {total_in:,}  kept: {total_kept:,}")

    if not chunks:
        return pd.DataFrame(columns=keep_cols)

    ratings = pd.concat(chunks, ignore_index=True)
    print(f"  final ratings rows: {len(ratings):,}")
    return ratings


def write_outputs(
    notes_final: pd.DataFrame,
    ratings: pd.DataFrame,
    notes_out: Path,
    ratings_out: Path,
) -> None:
    banner("STEP 8: Save outputs")
    notes_to_save = notes_final[
        ["noteId", "tweetId", "summary", "full_text", "writer", "createdAtMillis"]
    ].rename(columns={"summary": "note_text", "full_text": "post_text"})
    notes_to_save.to_parquet(notes_out, index=False)
    print(f"  wrote {notes_out}  ({len(notes_to_save):,} rows)")

    ratings.to_parquet(ratings_out, index=False)
    print(f"  wrote {ratings_out}  ({len(ratings):,} rows)")


def report(notes_final: pd.DataFrame, ratings: pd.DataFrame, summary_out: Path) -> None:
    banner("STEP 9: Summary report")
    note_writer = notes_final[["noteId", "tweetId", "writer"]]

    n_ai = (note_writer["writer"] == "ai").sum()
    n_human = (note_writer["writer"] == "human").sum()
    n_unique_posts = note_writer["tweetId"].nunique()

    ratings_with_meta = ratings.merge(note_writer, on="noteId", how="inner")
    n_ratings_human = (ratings_with_meta["writer"] == "human").sum()
    n_ratings_ai = (ratings_with_meta["writer"] == "ai").sum()
    n_unique_raters = ratings_with_meta["raterParticipantId"].nunique()

    grouped = ratings_with_meta.groupby(
        ["raterParticipantId", "tweetId"], dropna=False
    ).agg(
        n_notes=("noteId", "nunique"),
        n_levels=("rating", "nunique"),
    )
    n_pairwise = ((grouped["n_notes"] >= 2) & (grouped["n_levels"] >= 2)).sum()

    lines = [
        f"Unique AI notes:             {n_ai:,}",
        f"Unique human notes:          {n_human:,}",
        f"Unique notes total:          {n_ai + n_human:,}",
        f"Unique posts (tweetId):      {n_unique_posts:,}",
        f"Unique raters with ratings:  {n_unique_raters:,}",
        f"Ratings on human notes:      {n_ratings_human:,}",
        f"Ratings on AI notes:         {n_ratings_ai:,}",
        f"Pairwise eval candidates:    {n_pairwise:,}",
        "  (rater,post) groups where rater rated >=2 distinct notes with >=2 distinct rating levels",
    ]
    text = "\n".join(lines)
    print(text)
    summary_out.write_text(text + "\n")
    print(f"\n  wrote {summary_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notes-glob", required=True,
                   help="glob for Community Notes notes TSV shards")
    p.add_argument("--posts", nargs="+", required=True,
                   help="one or more post parquet files (unioned, dedup on tweetId)")
    p.add_argument("--user-enrollment", required=True,
                   help="Community Notes userEnrollment TSV (drives AI/human writer labeling)")
    p.add_argument("--rater-params", required=True,
                   help="rater params TSV; raters with non-null internalRaterIntercept are kept")
    p.add_argument("--ratings-glob", required=True,
                   help="glob for Community Notes ratings TSV shards")
    p.add_argument("--out-dir", required=True,
                   help="output directory for notes.parquet / ratings.parquet / summary.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_out = out_dir / "notes.parquet"
    ratings_out = out_dir / "ratings.parquet"
    summary_out = out_dir / "summary.txt"

    notes = load_notes(args.notes_glob)
    notes = label_writer(notes, args.user_enrollment)
    posts = load_posts(args.posts)
    merged = join_notes_posts(notes, posts)
    merged = filter_english_notes(merged)

    note_ids: Set[int] = set(merged["noteId"].dropna().astype("int64").tolist())
    print(f"\n  target noteIds: {len(note_ids):,}")

    valid_raters = load_valid_raters(args.rater_params)
    ratings = load_ratings(note_ids, valid_raters, args.ratings_glob)
    ratings = ratings.rename(columns={"helpfulnessLevel": "rating"})

    write_outputs(merged, ratings, notes_out, ratings_out)
    report(merged, ratings, summary_out)


if __name__ == "__main__":
    main()
