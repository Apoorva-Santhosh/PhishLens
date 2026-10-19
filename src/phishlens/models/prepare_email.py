"""Prepare six local source corpora, retaining provenance and discarding ambiguous examples."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from phishlens.config import ROOT
from phishlens.features.email_text import text_key

SOURCES = ("CEAS_08", "Enron", "Ling", "Nazario", "Nigerian_Fraud", "SpamAssasin")


def prepare(directory: Path) -> dict:
    """Keep subject/body only; omit sender, recipient, dates and supplied URL flags."""
    frames = []
    provenance = []
    for source in SOURCES:
        path = directory / f"{source}.csv"
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not {"subject", "body", "label"}.issubset(raw.columns):
            raise ValueError(f"{path.name} requires subject, body, label")
        if not raw.label.isin(["0", "1"]).all():
            raise ValueError(f"{path.name} labels must be 0 (legitimate), 1 (spam/phishing)")
        provenance.append(
            {
                "file": path.name,
                "rows": len(raw),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        frames.append(
            pd.DataFrame(
                {
                    "text": raw.subject + "\n" + raw.body,
                    "label": raw.label.astype(int),
                    "source": source,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    rejected: dict[str, int] = {}
    keys = []
    for text in frame.text:
        try:
            keys.append(text_key(text))
        except ValueError as exc:
            rejected[str(exc)] = rejected.get(str(exc), 0) + 1
            keys.append(None)
    frame["key"] = keys
    frame = frame.dropna(subset=["key"])
    conflicts = frame.groupby("key").label.nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    ambiguous_rows = int(frame.key.isin(conflict_keys).sum())
    frame = frame[~frame.key.isin(conflict_keys)]
    duplicates = int(frame.key.duplicated().sum())
    frame = frame.drop_duplicates("key").drop(columns="key")
    destination = directory / "emails.csv"
    frame.to_csv(destination, index=False)
    report = {
        "sources": provenance,
        "label_meaning": "0 legitimate, 1 spam/phishing proxy",
        "rejected": rejected,
        "conflicting_normalized_rows_removed": ambiguous_rows,
        "duplicate_normalized_rows_removed": duplicates,
        "rows": len(frame),
        "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()},
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    (directory / "emails.provenance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    """Prepare downloaded files without credentials or network calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data")
    print(json.dumps(prepare(parser.parse_args().directory), indent=2))


if __name__ == "__main__":
    main()
