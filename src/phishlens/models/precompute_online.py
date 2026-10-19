"""Slow, bounded, resumable acquisition: network collection is separate from model training."""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

from phishlens.config import DATA_PATH, ONLINE_MAX_DOMAINS, ONLINE_SNAPSHOT_PATH, SEED
from phishlens.features.url_online import OnlineLookup, OnlineSettings, UnsafeTarget, safe_domain


def collect(
    data: Path,
    output: Path,
    lookup: OnlineLookup,
    max_domains: int = ONLINE_MAX_DOMAINS,
    refresh: bool = False,
) -> dict:
    """Visit each sampled registered domain once, retaining completed rows on restart."""
    if max_domains < 1:
        raise ValueError("max_domains must be positive")
    frame = pd.read_csv(data, dtype=str, keep_default_na=False, usecols=["url", "label"])
    if not frame.label.isin(["0", "1"]).all():
        raise ValueError("URL labels must be 0 or 1")
    rows = []
    skipped = 0
    for url, label in zip(frame.url, frame.label, strict=True):
        try:
            rows.append((safe_domain(url), label))
        except ValueError:
            skipped += 1
    domains = pd.DataFrame(rows, columns=["domain", "label"]).drop_duplicates("domain")
    # Alternate shuffled classes so small bounded runs do not collect only popular benign domains.
    classes = [
        group.sample(frac=1, random_state=SEED).domain.tolist()
        for _, group in domains.groupby("label")
    ]
    selected = []
    for i in range(max(map(len, classes), default=0)):
        for group in classes:
            if i < len(group):
                selected.append(group[i])
    selected = selected[:max_domains]
    completed = lookup.completed_domains() if not refresh else set()
    counts = {"ok": 0, "failed": 0, "unsafe": 0, "resumed": 0}
    for domain in selected:
        if domain in completed:
            counts["resumed"] += 1
            continue
        started = time.monotonic()
        try:
            result = lookup.lookup(domain)
            counts["ok" if result.complete else "failed"] += 1
        except UnsafeTarget:
            counts["unsafe"] += 1
        # An atomic export after every domain makes interruption/resume safe.
        lookup.export(output)
        remaining = lookup.settings.min_interval - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
    lookup.export(output)
    report = {
        "selected_domains": len(selected),
        "invalid_or_unsafe_url_rows_skipped": skipped,
        "counts": counts,
        "snapshot": str(output),
        "note": "Current domain metadata may not describe historical URL labels.",
    }
    output.with_suffix(".collection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    """Require an explicit bounded run rather than silently crawling every dataset domain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ONLINE_SNAPSHOT_PATH)
    parser.add_argument("--max-domains", type=int, default=ONLINE_MAX_DOMAINS)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        report = collect(
            args.data, args.output, OnlineLookup(OnlineSettings()), args.max_domains, args.refresh
        )
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed cache/snapshot rows are retained.\n")
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.exit(1, f"Collection failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
