"""Compare saved models on fixed probes and the replacement's held-out domains."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from phishlens.config import ARTIFACT_DIR, DATA_PATH, SUSPICIOUS_CUTOFF
from phishlens.explain.url_shap import URLExplainer
from phishlens.features.url_offline import (
    FEATURE_NAMES,
    extract_features,
    parse_url,
    registered_domain,
)
from phishlens.models.data import load_urls, split_domains
from phishlens.models.evaluate import evaluate
from phishlens.models.prepare_url import SANITY_URLS
from phishlens.models.registry import load_artifact


def audit(data: Path, baseline: Path, candidate: Path, output: Path) -> dict:
    """Use probes for diagnosis only; model selection remains validation ROC-AUC."""
    old = load_artifact(baseline)
    new = load_artifact(candidate)
    if new["metadata"]["data_sha256"] != hashlib.sha256(data.read_bytes()).hexdigest():
        raise ValueError("Candidate was trained on a different data file")
    parts = split_domains(load_urls(data))
    all_domains = set().union(*(set(part.domain) for part in parts.values()))
    if any(registered_domain(url) in all_domains for url in SANITY_URLS):
        raise ValueError("Sanity-check domains must be excluded before training")
    for a, b in combinations(parts.values(), 2):
        if not set(a.domain).isdisjoint(b.domain):
            raise ValueError("URL partitions share registered domains")
    test = parts["test"]
    x = np.array([list(extract_features(url).values()) for url in test.url])
    old_prob = old["model"].predict_proba(x)[:, 1]
    new_prob = new["model"].predict_proba(x)[:, 1]
    deep = (test.label.to_numpy() == 0) & np.array(
        [parse_url(url).path not in ("", "/") for url in test.url]
    )
    explainer = URLExplainer.from_artifact(candidate)
    sanity = []
    for url in SANITY_URLS:
        features = extract_features(url)
        row = np.array([[features[name] for name in FEATURE_NAMES]])
        result = explainer.explain(url)
        assert result == explainer.explain(url)
        sanity.append(
            {
                "url": url,
                "old_probability": float(old["model"].predict_proba(row)[0, 1]),
                "new_probability": float(new["model"].predict_proba(row)[0, 1]),
                "top_features": result.top_features,
            }
        )
    target = sanity[0]
    report = {
        "candidate_model": new["metadata"]["model_type"],
        "selection": "Validation ROC-AUC then recall; probes were not used for selection.",
        "new_test_metrics": evaluate(test.label.to_numpy(), new_prob),
        "test_domains_disjoint": True,
        "legitimate_deep_link_test_count": int(deep.sum()),
        "old_deep_link_false_positive_rate_at_0_5": float((old_prob[deep] >= 0.5).mean()),
        "new_deep_link_false_positive_rate_at_0_5": float((new_prob[deep] >= 0.5).mean()),
        "probe_domains_excluded_from_dataset": True,
        "sanity_probes": sanity,
        "original_example_now_below_suspicious_cutoff": (
            target["new_probability"] < SUSPICIOUS_CUTOFF
        ),
        "sanity_probabilities_finite": bool(
            np.isfinite([r["new_probability"] for r in sanity]).all()
        ),
        "caveat": (
            "Probe labels are benign URL patterns, not live page verification. Old-model cohort "
            "comparison is diagnostic: its historical fit may overlap these domains. "
            "New test metrics are within-dataset, not independent temporal evaluation."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sanity_probes"}, indent=2))
    print(
        json.dumps([{k: v for k, v in r.items() if k != "top_features"} for r in sanity], indent=2)
    )
    return report


def main() -> None:
    """Audit existing local artifacts without fitting on diagnostic probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--baseline", type=Path, default=ARTIFACT_DIR / "phiusiil_baseline/url_offline.joblib"
    )
    parser.add_argument(
        "--candidate", type=Path, default=ARTIFACT_DIR / "url_candidate/url_offline.joblib"
    )
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "url_bias_audit.json")
    args = parser.parse_args()
    audit(args.data, args.baseline, args.candidate, args.output)


if __name__ == "__main__":
    main()
