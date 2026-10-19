"""Train only from a saved feature snapshot; this module never performs a network lookup."""

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phishlens import __version__
from phishlens.config import ARTIFACT_DIR, DATA_PATH, ONLINE_SNAPSHOT_PATH, SEED
from phishlens.features.url_offline import FEATURE_DESCRIPTIONS, FEATURE_NAMES, extract_features
from phishlens.features.url_online import ONLINE_DESCRIPTIONS, ONLINE_NAMES, safe_domain
from phishlens.models.data import load_urls, split_domains
from phishlens.models.evaluate import evaluate
from phishlens.models.registry import save_artifact
from phishlens.models.train_url import fit_models


def load_snapshot(path: Path) -> pd.DataFrame:
    """Accept complete successful rows only, rejecting ambiguous or corrupt domain snapshots."""
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"domain", "status", "observed_at", *ONLINE_NAMES}
    if not required.issubset(frame.columns):
        raise ValueError(f"Snapshot requires columns: {', '.join(sorted(required))}")
    if frame.domain.duplicated().any():
        raise ValueError("Snapshot must have one row per registered domain")
    for domain in frame.domain:
        if safe_domain(domain) != domain:
            raise ValueError("Snapshot keys must be registered domains")
    frame = frame.loc[frame.status == "ok"].copy()
    for name in (*ONLINE_NAMES, "observed_at"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame[[*ONLINE_NAMES, "observed_at"]].to_numpy(dtype=float)).all():
        raise ValueError("Successful snapshot rows must have finite, complete features")
    if (frame[[*ONLINE_NAMES, "observed_at"]] < 0).any().any():
        raise ValueError("Snapshot values must be nonnegative")
    if not frame.registrar_present.isin([0, 1]).all():
        raise ValueError("registrar_present must be 0 or 1")
    if frame.empty:
        raise ValueError("No complete successful snapshots: collect WHOIS/DNS features first")
    return frame


def train(data_path: Path, snapshot_path: Path, output: Path) -> dict:
    """Join frozen domain features, group by queried domain, calibrate and register separately."""
    source = load_urls(data_path)
    snapshot = load_snapshot(snapshot_path)
    keys = []
    for url in source.url:
        try:
            keys.append(safe_domain(url))
        except ValueError:
            keys.append(None)
    source["domain"] = keys
    frame = source.merge(snapshot, on="domain", how="inner", validate="many_to_one")
    parts = split_domains(frame)
    names = (*FEATURE_NAMES, *ONLINE_NAMES)
    arrays = {}
    for name, part in parts.items():
        lexical = np.asarray([list(extract_features(url).values()) for url in part.url])
        extended = part[list(ONLINE_NAMES)].to_numpy(dtype=float)
        arrays[name] = (np.column_stack((lexical, extended)), part.label.to_numpy())
    model, winner, validation = fit_models(arrays)
    metrics = evaluate(arrays["test"][1], model.predict_proba(arrays["test"][0])[:, 1])
    metadata = {
        "version": f"{__version__}-online-pilot",
        "experimental": True,
        "mode": "online",
        "model_type": winner,
        "seed": SEED,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_names": list(names),
        "feature_descriptions": {**FEATURE_DESCRIPTIONS, **ONLINE_DESCRIPTIONS},
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "rows_with_complete_snapshots": len(frame),
        "source_rows": len(source),
        "successful_snapshot_domains": len(snapshot),
        "source_coverage_fraction": len(frame) / len(source),
        "partitions": {
            n: {
                "rows": len(p),
                "domains": p.domain.nunique(),
                "class_counts": {str(k): int(v) for k, v in p.label.value_counts().items()},
            }
            for n, p in parts.items()
        },
        "validation_metrics": validation,
        "test_metrics": metrics,
        "selection_reason": f"{winner} maximizes validation ROC-AUC, then phishing recall.",
        "calibration": "Independent sigmoid on frozen estimator; no final refit.",
        "lookup_grouping": "ICANN registered domain, including shared private-suffix tenants.",
        "limitation": "Lookup-success selection bias; present metadata with historic labels.",
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scikit-learn", "xgboost", "dnspython", "python-whois")
        },
    }
    save_artifact(output, model, metadata, name="url_online")
    (output / "url_online_report.md").write_text(
        "# Online URL model report\n\nExperimental pilot; not production validated.\n\n"
        + metadata["selection_reason"]
        + "\n\nThreshold 0.5; matrices [[TN, FP], [FN, TP]].\n\n```json\n"
        + json.dumps({"validation": validation, "test": metrics}, indent=2)
        + "\n```\n\n"
        + metadata["limitation"]
        + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """Train from files only; fail clearly when acquisition has not supplied enough domains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--snapshot", type=Path, default=ONLINE_SNAPSHOT_PATH)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.snapshot, args.output), indent=2))


if __name__ == "__main__":
    main()
