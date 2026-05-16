#!/usr/bin/env python3
"""Convert HateXplain.json to a processed CSV with text and majority-vote labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def majority_class_label(annotators: list[dict]) -> str:
    """Three-class majority label (hatespeech / offensive / normal / undecided)."""
    counts = Counter(a["label"] for a in annotators)
    label, votes = counts.most_common(1)[0]
    return label if votes >= 2 else "undecided"


def to_binary_label(class_label: str) -> str:
    """Map HateXplain classes to hateful vs non-hateful (toxic / non-toxic)."""
    if class_label in ("hatespeech", "offensive"):
        return "hateful"
    return "non-hateful"


def majority_targets(annotators: list[dict], class_label: str) -> list[str]:
    if class_label not in ("hatespeech", "offensive"):
        return []
    counts: Counter[str] = Counter()
    for ann in annotators:
        for target in ann.get("target", []):
            if target and target != "None":
                counts[target] += 1
    return sorted(t for t, n in counts.items() if n >= 2)


def tokens_to_text(post_tokens: list[str]) -> str:
    return " ".join(post_tokens)


def sorted_annotators(annotators: list[dict]) -> list[dict]:
    return sorted(annotators, key=lambda a: a["annotator_id"])


def source_from_post_id(post_id: str) -> str:
    if post_id.endswith("_twitter"):
        return "twitter"
    if post_id.endswith("_gab"):
        return "gab"
    return post_id.rsplit("_", 1)[-1] if "_" in post_id else "unknown"


def has_rationale(rationales: list, class_label: str) -> bool:
    if class_label not in ("hatespeech", "offensive"):
        return False
    return bool(rationales)


def convert(input_path: Path, output_path: Path) -> None:
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    fieldnames = [
        "post_id",
        "source",
        "text",
        "final_label",
        "majority_class",
        "label_annotator_1",
        "label_annotator_2",
        "label_annotator_3",
        "target_annotator_1",
        "target_annotator_2",
        "target_annotator_3",
        "final_targets",
        "token_count",
        "has_rationale",
    ]

    rows = []
    for post_id, record in data.items():
        annotators = sorted_annotators(record["annotators"])
        majority_class = majority_class_label(annotators)
        final_label = to_binary_label(majority_class)
        targets_final = majority_targets(annotators, majority_class)

        def fmt_targets(ann: dict) -> str:
            return "|".join(t for t in ann.get("target", []) if t)

        rows.append(
            {
                "post_id": post_id,
                "source": source_from_post_id(post_id),
                "text": tokens_to_text(record["post_tokens"]),
                "final_label": final_label,
                "majority_class": majority_class,
                "label_annotator_1": annotators[0]["label"],
                "label_annotator_2": annotators[1]["label"],
                "label_annotator_3": annotators[2]["label"],
                "target_annotator_1": fmt_targets(annotators[0]),
                "target_annotator_2": fmt_targets(annotators[1]),
                "target_annotator_3": fmt_targets(annotators[2]),
                "final_targets": "|".join(targets_final),
                "token_count": len(record["post_tokens"]),
                "has_rationale": has_rationale(record.get("rationales", []), majority_class),
            }
        )

    rows.sort(key=lambda r: r["post_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(r["final_label"] for r in rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    print("final_label distribution:", dict(label_counts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "HateXplain.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "HateXplain_processed.csv",
    )
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
