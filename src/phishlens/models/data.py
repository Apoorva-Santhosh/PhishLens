"""Validate local CSVs and prevent registered-domain leakage."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from phishlens.config import SEED
from phishlens.features.url_offline import parse_url, registered_domain


def load_urls(path: str | Path) -> pd.DataFrame:
    """Reject ambiguous labels and conflicting duplicates rather than guessing."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"url", "label"}.issubset(frame.columns):
        raise ValueError("CSV requires columns: url, label (0=legitimate, 1=phishing)")
    frame = frame[["url", "label"]].copy()
    frame["url"] = frame.url.str.strip()
    frame["label"] = frame.label.str.strip()
    if not frame.label.isin(["0", "1"]).all():
        raise ValueError("Labels must be exactly 0 (legitimate) or 1 (phishing)")
    for row, url in enumerate(frame.url, start=2):
        try:
            parse_url(url)
        except ValueError as exc:
            raise ValueError(f"CSV row {row}: {exc}") from exc
    if (frame.groupby("url").label.nunique() > 1).any():
        raise ValueError("Conflicting labels for duplicate URLs")
    frame = frame.drop_duplicates("url").reset_index(drop=True)
    frame["label"] = frame.label.astype(int)
    if set(frame.label) != {0, 1}:
        raise ValueError("Dataset must contain both classes")
    frame["domain"] = frame.url.map(registered_domain)
    return frame


def split_domains(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reserve 20% test, 20% validation, 15% calibration, and 45% fit groups."""
    remaining = frame
    partitions = {}
    for name, fraction in [("test", 0.2), ("validation", 0.25), ("calibration", 0.25)]:
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=SEED)
        try:
            keep, hold = next(splitter.split(remaining, groups=remaining.domain))
        except ValueError as exc:
            raise ValueError("Too few distinct domains for four partitions") from exc
        partitions[name] = remaining.iloc[hold].copy()
        remaining = remaining.iloc[keep].copy()
    partitions["train"] = remaining
    for name, part in partitions.items():
        if set(part.label) != {0, 1}:
            raise ValueError(f"{name} partition needs both classes; supply more diverse domains")
    return partitions
