"""Explain the base estimator; nonlinear calibration is not SHAP-additive."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from phishlens.config import ARTIFACT_DIR
from phishlens.features.url_offline import FEATURE_DESCRIPTIONS, FEATURE_NAMES, extract_features
from phishlens.features.url_online import ONLINE_DESCRIPTIONS, ONLINE_NAMES
from phishlens.models.registry import load_artifact

EXPLANATION_NOTE = (
    "SHAP explains the underlying fitted model, not its calibrated probability or risk score. "
    "Contributions are oriented toward phishing risk using the sigmoid calibration direction. "
    "Contributions, baseline and explained value are multiplied by calibration_orientation "
    "(-1 for decreasing, +1 for increasing, 0 for constant calibration). "
    "Positive values raise risk relative to the model baseline; negative values lower it. "
    "Only all feature contributions together reconstruct explained_value from base_value."
)


class FeatureContribution(TypedDict):
    """Keep the explanation payload ready for the API and dashboard."""

    name: str
    value: float
    contribution: float
    direction: Literal["raises_risk", "lowers_risk", "neutral"]
    description: str


@dataclass(frozen=True)
class URLExplanation:
    """Declare the units and baseline so magnitudes cannot masquerade as risk points."""

    top_features: list[FeatureContribution]
    base_value: float
    explained_value: float
    output_space: str
    calibration_orientation: int
    note: str = EXPLANATION_NOTE


class URLExplainer:
    """Build once per loaded model, reusing deterministic exact SHAP explainers."""

    def __init__(
        self, model: CalibratedClassifierCV, feature_names: tuple[str, ...] = FEATURE_NAMES
    ) -> None:
        """Support Stage 1's single frozen estimator with binary sigmoid calibration."""
        if not isinstance(model, CalibratedClassifierCV) or not hasattr(
            model, "calibrated_classifiers_"
        ):
            raise ValueError("Expected a fitted CalibratedClassifierCV from Stage 1")
        if model.method != "sigmoid" or len(model.calibrated_classifiers_) != 1:
            raise ValueError("URL explanations require a single sigmoid-calibrated estimator")
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError("URL model classes must be [0, 1], with 1 meaning phishing")
        calibrated = model.calibrated_classifiers_[0]
        base = calibrated.estimator
        self.estimator = base.estimator if isinstance(base, FrozenEstimator) else base
        if feature_names not in {FEATURE_NAMES, (*FEATURE_NAMES, *ONLINE_NAMES)}:
            raise ValueError("Unsupported URL feature schema")
        self.feature_names = feature_names
        self.descriptions = {**FEATURE_DESCRIPTIONS, **ONLINE_DESCRIPTIONS}
        if self.estimator.n_features_in_ != len(feature_names):
            raise ValueError("Model feature count does not match the URL schema")
        # The fitted sigmoid is expit(-(a * score + b)). It can reverse direction.
        # Orient SHAP signs accordingly, but do not claim additive calibrated probabilities.
        self.orientation = int(np.sign(-calibrated.calibrators[0].a_))
        self.scaler: StandardScaler | None = None
        if isinstance(self.estimator, Pipeline):
            steps = self.estimator.steps
            if (
                len(steps) != 2
                or not isinstance(steps[0][1], StandardScaler)
                or not isinstance(steps[1][1], LogisticRegression)
                or not steps[0][1].with_mean
            ):
                raise ValueError(
                    "Expected Stage 1's StandardScaler and logistic regression pipeline"
                )
            self.scaler = steps[0][1]
            # Training means become zeros after centering; no calibration/test rows are used.
            background = np.zeros((1, len(self.feature_names)))
            self._shap: Any = shap.LinearExplainer(steps[1][1], background)
            self.output_space = "base_model_log_odds"
        elif isinstance(self.estimator, (RandomForestClassifier, XGBClassifier)):
            self._shap = shap.TreeExplainer(
                self.estimator, feature_perturbation="tree_path_dependent", model_output="raw"
            )
            self.output_space = (
                "base_model_probability"
                if isinstance(self.estimator, RandomForestClassifier)
                else "base_model_log_odds"
            )
        else:
            raise ValueError(
                "Supported URL estimators are logistic regression, forest, and XGBoost"
            )
        if self.orientation != 1:
            self.output_space = "oriented_" + self.output_space

    @classmethod
    def from_artifact(cls, path: Path) -> "URLExplainer":
        """Load a trusted registered model and reject a mismatched feature order."""
        artifact = load_artifact(path)
        names = tuple(artifact["metadata"].get("feature_names", []))
        if names not in {FEATURE_NAMES, (*FEATURE_NAMES, *ONLINE_NAMES)}:
            raise ValueError("Artifact feature schema does not match known URL features")
        return cls(artifact["model"], names)

    def explain(
        self, url: str, top_k: int = 8, online_features: dict[str, float | None] | None = None
    ) -> URLExplanation:
        """Rank by absolute contribution; ties retain schema order (default top eight)."""
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= len(self.feature_names)
        ):
            raise ValueError(f"top_k must be an integer between 1 and {len(self.feature_names)}")
        features = extract_features(url)
        if self.feature_names != FEATURE_NAMES:
            if online_features is None or set(online_features) != set(ONLINE_NAMES):
                raise ValueError(
                    "Online URL explanations require a complete precomputed feature set"
                )
            try:
                if not all(
                    value is not None and np.isfinite(value) for value in online_features.values()
                ):
                    raise ValueError("Missing or invalid online explanation features")
            except TypeError as exc:
                raise ValueError("Online explanation features must be finite numbers") from exc
            features.update(online_features)
        raw = np.asarray([[features[name] for name in self.feature_names]])
        transformed = self.scaler.transform(raw) if self.scaler is not None else raw
        explanation = self._shap(transformed)
        values = np.asarray(explanation.values, dtype=float)
        baselines = np.asarray(explanation.base_values, dtype=float)
        # Forest SHAP has one output per class; select class 1, never predicted class.
        if values.ndim == 3:
            contributions = values[0, :, 1]
            baseline = float(baselines[0, 1])
        else:
            contributions = values[0]
            baseline = float(baselines.reshape(-1)[0])
        if (
            contributions.shape != (len(self.feature_names),)
            or not np.isfinite(contributions).all()
        ):
            raise ValueError("SHAP returned invalid URL contributions")
        contributions = contributions * self.orientation
        baseline *= self.orientation
        ranked = sorted(range(len(self.feature_names)), key=lambda i: -abs(contributions[i]))
        top_features: list[FeatureContribution] = []
        for index in ranked[:top_k]:
            name = self.feature_names[index]
            contribution = float(contributions[index])
            direction: Literal["raises_risk", "lowers_risk", "neutral"] = "neutral"
            if contribution > 0:
                direction = "raises_risk"
            elif contribution < 0:
                direction = "lowers_risk"
            top_features.append(
                {
                    "name": name,
                    "value": features[name],
                    "contribution": contribution,
                    "direction": direction,
                    "description": self.descriptions[name],
                }
            )
        return URLExplanation(
            top_features=top_features,
            base_value=baseline,
            explained_value=baseline + float(contributions.sum()),
            output_space=self.output_space,
            calibration_orientation=self.orientation,
        )


def main() -> None:
    """Inspect an explanation from a trusted local artifact without any URL lookup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_DIR / "url_offline.joblib")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    result = URLExplainer.from_artifact(args.artifact).explain(args.url, args.top_k)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
