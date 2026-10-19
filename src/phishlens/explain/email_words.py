"""Exact present-term contributions; offsets refer to cleaned text, never raw HTML."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishlens.features.email_text import clean_text
from phishlens.models.registry import load_artifact


@dataclass(frozen=True)
class EmailExplanation:
    """Expose exact raw contributions separately from risk-directed display contributions."""

    top_features: list[dict[str, Any]]
    highlights: list[dict[str, Any]]
    cleaned_text: str
    base_value: float
    explained_value: float
    calibration_orientation: int
    note: str = (
        "Raw coefficient-times-TF-IDF contributions add to base-model log-odds. "
        "Display contributions are oriented by sigmoid calibration, not calibrated probability. "
        "Highlights refer to cleaned_text; overlapping terms are not extra contributions."
    )


class EmailExplainer:
    """Reuse one fitted vectorizer and classifier without refitting or densifying documents."""

    def __init__(self, model: CalibratedClassifierCV) -> None:
        """Accept the single sigmoid-calibrated pipeline produced by email training."""
        if not isinstance(model, CalibratedClassifierCV) or not hasattr(
            model, "calibrated_classifiers_"
        ):
            raise ValueError("Expected a fitted calibrated email model")
        if model.method != "sigmoid" or len(model.calibrated_classifiers_) != 1:
            raise ValueError("Email explanations require one sigmoid-calibrated estimator")
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError("Email classes must be [0, 1]")
        calibrated = model.calibrated_classifiers_[0]
        base = calibrated.estimator
        self.pipeline = base.estimator if isinstance(base, FrozenEstimator) else base
        if not isinstance(self.pipeline, Pipeline) or not isinstance(
            self.pipeline.named_steps.get("classifier"), LogisticRegression
        ):
            raise ValueError("Email model must contain TF-IDF and logistic regression")
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.names = self.vectorizer.get_feature_names_out()
        self.orientation = int(np.sign(-calibrated.calibrators[0].a_))

    @classmethod
    def from_artifact(cls, path: Path) -> "EmailExplainer":
        """Check the vocabulary against metadata when loading a trusted artifact."""
        artifact = load_artifact(path)
        result = cls(artifact["model"])
        if artifact["metadata"].get("feature_names") != result.names.tolist():
            raise ValueError("Email artifact vocabulary does not match metadata")
        return result

    def explain(self, text: str, top_k: int = 8) -> EmailExplanation:
        """Include both risk-raising and lowering present terms when both are available."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 2:
            raise ValueError("top_k must be an integer of at least 2")
        cleaned = clean_text(text)
        row = self.vectorizer.transform([text]).tocsr()
        raw = self.classifier.coef_[0, row.indices] * row.data
        display = raw * self.orientation
        ranked = sorted(
            range(len(raw)), key=lambda i: (-abs(display[i]), self.names[row.indices[i]])
        )
        selected = ranked[:top_k]
        # Reserve a slot for the other direction instead of hiding all mitigating terms.
        for sign in (1, -1):
            candidate = next((i for i in ranked if display[i] * sign > 0), None)
            if candidate is not None and not any(display[i] * sign > 0 for i in selected):
                selected[-1] = candidate
        selected = sorted(
            set(selected), key=lambda i: (-abs(display[i]), self.names[row.indices[i]])
        )
        features = []
        highlights = []
        tokens = list(re.finditer(r"(?u)\b\w\w+\b", cleaned))
        for i in selected:
            term = str(self.names[row.indices[i]])
            contribution = float(display[i])
            direction = (
                "raises_risk"
                if contribution > 0
                else "lowers_risk"
                if contribution < 0
                else "neutral"
            )
            features.append(
                {
                    "name": term,
                    "value": float(row.data[i]),
                    "contribution": contribution,
                    "raw_contribution": float(raw[i]),
                    "direction": direction,
                    "description": "Logistic coefficient multiplied by this term's TF-IDF.",
                }
            )
            words = term.split()
            for offset in range(len(tokens) - len(words) + 1):
                window = tokens[offset : offset + len(words)]
                if [token.group() for token in window] != words:
                    continue
                highlights.append(
                    {
                        "term": term,
                        "start": window[0].start(),
                        "end": window[-1].end(),
                        "direction": direction,
                        "contribution": contribution,
                    }
                )
        baseline = float(self.classifier.intercept_[0])
        return EmailExplanation(
            features, highlights, cleaned, baseline, baseline + float(raw.sum()), self.orientation
        )
