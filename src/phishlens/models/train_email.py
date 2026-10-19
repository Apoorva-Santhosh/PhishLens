"""Train a calibrated TF-IDF/logistic model on deduplicated local text."""

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from phishlens import __version__
from phishlens.config import ARTIFACT_DIR, EMAIL_DATA_PATH, MAX_EMAIL_LENGTH, SEED
from phishlens.features.email_text import build_pipeline, text_key
from phishlens.models.evaluate import evaluate
from phishlens.models.registry import save_artifact


def load_emails(path: Path) -> pd.DataFrame:
    """Validate the generic text/label contract and eliminate normalized duplicate leakage."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Email CSV requires text and label columns")
    if not frame.label.isin(["0", "1"]).all():
        raise ValueError("Email labels must be 0 (legitimate) or 1 (spam/phishing)")
    keys = []
    for row, text in enumerate(frame.text, start=2):
        try:
            keys.append(text_key(text))
        except ValueError as exc:
            raise ValueError(f"Email CSV row {row}: {exc}") from exc
    frame["key"] = keys
    if (frame.groupby("key").label.nunique() > 1).any():
        raise ValueError("Conflicting labels for normalized email text")
    frame = frame.drop_duplicates("key").reset_index(drop=True)
    frame["label"] = frame.label.astype(int)
    if set(frame.label) != {0, 1}:
        raise ValueError("Email dataset must contain both classes")
    return frame


def split_emails(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stratify disjoint normalized documents into fit/calibration/validation/test sets."""
    remaining = frame
    parts = {}
    try:
        for name, size in [("test", 0.2), ("validation", 0.25), ("calibration", 0.25)]:
            remaining, hold = train_test_split(
                remaining, test_size=size, stratify=remaining.label, random_state=SEED
            )
            parts[name] = hold.copy()
    except ValueError as exc:
        raise ValueError("Not enough examples per class for four stratified partitions") from exc
    parts["train"] = remaining.copy()
    return parts


def train(data_path: Path, output: Path) -> dict:
    """Keep both TF-IDF fitting and probability calibration away from final test data."""
    frame = load_emails(data_path)
    parts = split_emails(frame)
    pipeline = build_pipeline().fit(parts["train"].text, parts["train"].label)
    model = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    model.fit(parts["calibration"].text, parts["calibration"].label)
    metrics = {
        name: evaluate(part.label.to_numpy(), model.predict_proba(part.text)[:, 1])
        for name, part in parts.items()
        if name in {"validation", "test"}
    }
    metadata = {
        "version": __version__,
        "model_type": "tfidf_logistic_regression",
        "trained_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "rows_after_cleaning": len(frame),
        "max_input_length": MAX_EMAIL_LENGTH,
        "vocabulary_size": len(pipeline["tfidf"].vocabulary_),
        "feature_names": pipeline["tfidf"].get_feature_names_out().tolist(),
        "partitions": {
            name: {
                "rows": len(part),
                "class_counts": {str(k): int(v) for k, v in part.label.value_counts().items()},
            }
            for name, part in parts.items()
        },
        "validation_metrics": metrics["validation"],
        "test_metrics": metrics["test"],
        "calibration": "independent sigmoid; no final refit",
        "label_limitation": "Positive labels include spam and fraud, not exclusively phishing.",
        "explanation": "Exact coefficient times TF-IDF in base-model log-odds space.",
        "dependencies": {
            p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "joblib")
        },
    }
    save_artifact(output, model, metadata, name="email")
    (output / "email_report.md").write_text(
        "# Email model report\n\nTF-IDF + logistic regression, independent sigmoid calibration.\n\n"
        "Threshold: 0.5. Confusion matrices: [[TN, FP], [FN, TP]].\n\n```json\n"
        + json.dumps(metrics, indent=2)
        + "\n```\n\n"
        + "Positive labels include spam/fraud. Source, language and near-duplicate campaign bias "
        + "remain; this random document split is not an independent-source evaluation.\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """Train only from a prepared local CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=EMAIL_DATA_PATH)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    metadata = train(args.data, args.output)
    print(json.dumps({k: v for k, v in metadata.items() if k != "feature_names"}, indent=2))


if __name__ == "__main__":
    main()
