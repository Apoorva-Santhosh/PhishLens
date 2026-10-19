"""Compare calibrated models using validation only; evaluate the winner once."""

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from phishlens import __version__
from phishlens.config import ARTIFACT_DIR, DATA_PATH, SEED
from phishlens.features.url_offline import FEATURE_DESCRIPTIONS, FEATURE_NAMES, extract_features
from phishlens.models.data import load_urls, split_domains
from phishlens.models.evaluate import evaluate
from phishlens.models.registry import save_artifact


def fit_models(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[CalibratedClassifierCV, str, dict]:
    """Share fit/calibration/validation selection across offline and frozen-snapshot models."""
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=SEED,
            n_jobs=1,
            tree_method="hist",
            device="cpu",
        ),
    }
    validation = {}
    calibrated = {}
    for name, estimator in candidates.items():
        estimator.fit(*arrays["train"])
        model = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
        model.fit(*arrays["calibration"])
        calibrated[name] = model
        validation[name] = evaluate(
            arrays["validation"][1], model.predict_proba(arrays["validation"][0])[:, 1]
        )
    winner = max(validation, key=lambda n: (validation[n]["roc_auc"], validation[n]["recall"]))
    return calibrated[winner], winner, validation


def train(data_path: Path, output: Path) -> dict:
    """Fit on local files only, with independent calibration and selection sets."""
    frame = load_urls(data_path)
    parts = split_domains(frame)
    arrays = {
        name: (
            np.asarray([list(extract_features(url).values()) for url in part.url]),
            part.label.to_numpy(),
        )
        for name, part in parts.items()
    }
    model, winner, validation = fit_models(arrays)
    test_metrics = evaluate(arrays["test"][1], model.predict_proba(arrays["test"][0])[:, 1])
    metadata = {
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(),
        "model_type": winner,
        "feature_names": list(FEATURE_NAMES),
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "seed": SEED,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "rows_after_cleaning": len(frame),
        "partitions": {
            n: {"rows": len(p), "domains": p.domain.nunique()} for n, p in parts.items()
        },
        "validation_metrics": validation,
        "test_metrics": test_metrics,
        "selection_reason": (
            f"{winner} maximizes validation ROC-AUC, with phishing recall as tiebreaker. "
            "Final test data was used only after selection; no final refit was performed."
        ),
        "calibration": "sigmoid on independent registered-domain groups",
        "explanation": {
            "method": "LinearExplainer" if winner == "logistic_regression" else "TreeExplainer",
            "target_class": "phishing (1)",
            "calibrated_probability_explained": False,
            "note": "Contributions explain the base model, oriented by calibration direction.",
        },
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scikit-learn", "xgboost", "tldextract", "shap")
        },
    }
    save_artifact(output, model, metadata)
    report = [
        "# URL model report",
        "",
        metadata["selection_reason"],
        "",
        "Metrics use threshold 0.5; confusion matrices are [[TN, FP], [FN, TP]].",
        "",
        "```json",
        json.dumps({"validation": validation, "test": test_metrics}, indent=2),
        "```",
        "",
        "These results apply only to the supplied dataset and split.",
        "",
        "Explainability: SHAP explains the underlying fitted model, not calibrated probability.",
        "Signs point toward phishing risk; magnitudes are not 0-100 risk-score points.",
    ]
    (output / "url_report.md").write_text("\n".join(report), encoding="utf-8")
    return metadata


def main() -> None:
    """Expose reproducible training without a notebook dependency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
