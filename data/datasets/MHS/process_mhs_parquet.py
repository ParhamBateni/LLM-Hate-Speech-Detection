#!/usr/bin/env python3
"""Aggregate MHS parquet annotations to one row per comment with text and labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

HATE_SPEECH_SCORE_THRESHOLD = 0.5

MAJORITY_CLASS_MAP = {
    0.0: "not_hateful",
    1.0: "ambiguous",
    2.0: "hateful",
}


def majority_hatespeech(labels: pd.Series) -> float:
    return labels.value_counts().idxmax()


def to_binary_label(hate_speech_score: float) -> str:
    """MHS benchmark binary label (Kennedy et al. / DefVerify: score > 0.5)."""
    return "hateful" if hate_speech_score > HATE_SPEECH_SCORE_THRESHOLD else "non-hateful"


def to_majority_class(hatespeech_code: float) -> str:
    return MAJORITY_CLASS_MAP[float(hatespeech_code)]


def convert(input_path: Path, output_path: Path) -> None:
    df = pd.read_parquet(input_path)

    required = {"comment_id", "text", "hatespeech", "hate_speech_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    comments = df.drop_duplicates(subset="comment_id", keep="first")[
        ["comment_id", "text", "hate_speech_score", "platform"]
    ].copy()

    majority = (
        df.groupby("comment_id")["hatespeech"]
        .apply(majority_hatespeech)
        .rename("hatespeech_majority_code")
    )
    annotator_counts = df.groupby("comment_id")["annotator_id"].nunique().rename(
        "annotator_count"
    )

    comments = comments.merge(majority, on="comment_id").merge(
        annotator_counts, on="comment_id"
    )
    comments["majority_class"] = comments["hatespeech_majority_code"].map(
        to_majority_class
    )
    comments["final_label"] = comments["hate_speech_score"].map(to_binary_label)

    out = comments[
        [
            "comment_id",
            "text",
            "final_label",
            "majority_class",
            "hate_speech_score",
            "hatespeech_majority_code",
            "annotator_count",
            "platform",
        ]
    ].sort_values("comment_id")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Wrote {len(out)} rows to {output_path}")
    print("final_label:", out["final_label"].value_counts().to_dict())
    print("majority_class:", out["majority_class"].value_counts().to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "0000.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "MHS_processed.csv",
    )
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
