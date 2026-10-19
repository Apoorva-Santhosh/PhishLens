"""Convert downloaded Kaggle raw URLs without inventing paths or labels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from phishlens.config import ROOT
from phishlens.features.url_offline import parse_url, registered_domain

SANITY_URLS = (
    "https://example.com/about",
    "https://www.example.com/",
    "https://docs.python.org/3/library/pathlib.html",
    "https://python.org/about/",
    "https://www.python.org/downloads/",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://github.com/python/cpython/blob/main/README.rst",
)


def prepare(source: Path, destination: Path) -> dict:
    """Audit invalid/conflicting rows and preserve the source's good/bad mapping."""
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    if not {"URL", "Label"}.issubset(frame.columns):
        raise ValueError("Kaggle CSV requires URL and Label columns")
    if not frame.Label.isin(["good", "bad"]).all():
        raise ValueError("Kaggle labels must be good (0) or bad (1)")
    original_rows = len(frame)
    frame = pd.DataFrame(
        {"url": frame.URL.str.strip(), "label": frame.Label.map({"good": 0, "bad": 1})}
    )
    rejected: Counter[str] = Counter()
    valid = []
    for url in frame.url:
        try:
            parse_url(url)
            valid.append(True)
        except ValueError as exc:
            rejected[str(exc)] += 1
            valid.append(False)
    frame = frame.loc[valid]
    conflicts = frame.groupby("url").label.nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    conflict_rows = int(frame.url.isin(conflicting).sum())
    frame = frame[~frame.url.isin(conflicting)]
    duplicates = int(frame.url.duplicated().sum())
    frame = frame.drop_duplicates("url")
    reserved_domains = {registered_domain(url) for url in SANITY_URLS}
    reserved = frame.url.map(registered_domain).isin(reserved_domains)
    reserved_rows = int(reserved.sum())
    frame = frame[~reserved]
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    summary = {}
    for label, group in frame.groupby("label"):
        parsed = group.url.map(parse_url)
        summary[str(label)] = {
            "rows": len(group),
            "deep_path_fraction": float(parsed.map(lambda p: p.path not in ("", "/")).mean()),
            "www_fraction": float(parsed.map(lambda p: p.hostname.startswith("www.")).mean()),
            "explicit_scheme_fraction": float(
                group.url.map(lambda u: bool(urlsplit(u).scheme)).mean()
            ),
        }
    report = {
        "source": "https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls",
        "download_date": "2026-10-17",
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "original_rows": original_rows,
        "invalid_rows_by_reason": dict(rejected),
        "conflicting_rows_removed": conflict_rows,
        "duplicate_rows_removed": duplicates,
        "rows": len(frame),
        "label_mapping": {"good": 0, "bad": 1},
        "class_structure": summary,
        "sanity_domains_excluded": sorted(reserved_domains),
        "sanity_domain_rows_excluded": reserved_rows,
        "limitations": "Historical good/bad labels may be stale; no URLs are fetched.",
    }
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    """Prepare an already downloaded CSV; training never downloads files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/phishing_site_urls.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/urls.csv")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
