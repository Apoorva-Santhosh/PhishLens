"""One application entry point with explicit offline fallback and no web fetching."""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from phishlens.config import ARTIFACT_DIR
from phishlens.explain.email_words import EmailExplainer
from phishlens.explain.url_shap import URLExplainer
from phishlens.features.url_offline import FEATURE_NAMES, extract_features
from phishlens.features.url_online import ONLINE_NAMES, OnlineLookup, safe_domain
from phishlens.models.registry import load_artifact
from phishlens.scoring import score_probability


@dataclass(frozen=True)
class Prediction:
    """Keep API-ready results explicit about which trained model actually produced them."""

    risk_score: int
    label: str
    probability: float
    mode: str
    model_version: str
    top_features: list[dict[str, Any]]
    highlights: list[dict[str, Any]] = field(default_factory=list)
    cleaned_text: str | None = None
    fallback_reason: str | None = None


class Predictor:
    """Load artifacts once and lazily initialize expensive SHAP explainers."""

    def __init__(
        self,
        offline: dict[str, Any],
        online: dict[str, Any] | None = None,
        email: dict[str, Any] | None = None,
        lookup: OnlineLookup | None = None,
    ) -> None:
        """Validate schemas up front so an online model cannot receive offline columns."""
        if offline["metadata"].get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Offline artifact feature schema mismatch")
        if online is not None and online["metadata"].get("feature_names") != [
            *FEATURE_NAMES,
            *ONLINE_NAMES,
        ]:
            raise ValueError("Online artifact feature schema mismatch")
        self.offline = offline
        self.online = online
        self.email = email
        self.lookup = lookup or OnlineLookup()

    @classmethod
    def from_directory(cls, directory: Path = ARTIFACT_DIR) -> "Predictor":
        """Allow online/email artifacts to be absent while keeping offline URL prediction usable."""
        offline = load_artifact(directory / "url_offline.joblib")
        online_path = directory / "url_online.joblib"
        email_path = directory / "email.joblib"
        return cls(
            offline,
            load_artifact(online_path) if online_path.exists() else None,
            load_artifact(email_path) if email_path.exists() else None,
        )

    @cached_property
    def offline_explainer(self) -> URLExplainer:
        """Reuse one fitted explainer across offline and fallback predictions."""
        return URLExplainer(self.offline["model"])

    @cached_property
    def online_explainer(self) -> URLExplainer:
        """Explain extended features without performing a second network lookup."""
        if self.online is None:
            raise ValueError("Online model is not loaded")
        return URLExplainer(self.online["model"], (*FEATURE_NAMES, *ONLINE_NAMES))

    @cached_property
    def email_explainer(self) -> EmailExplainer:
        """Keep email explanations on the same fitted pipeline as probability prediction."""
        if self.email is None:
            raise ValueError("Email model is not loaded")
        return EmailExplainer(self.email["model"])

    def predict_url(self, url: str, mode: str = "offline") -> Prediction:
        """Reject unsafe online targets; fall back on incomplete metadata or missing models."""
        if mode not in {"offline", "online"}:
            raise ValueError("URL mode must be offline or online")
        features = extract_features(url)
        artifact = self.offline
        actual_mode = "offline"
        fallback_reason = None
        online_features = None
        if mode == "online":
            safe_domain(url)  # Reject private/reserved inputs even if no online model is loaded.
            if self.online is None:
                fallback_reason = "online_model_unavailable"
            else:
                result = self.lookup.lookup(url)
                if result.complete:
                    artifact = self.online
                    online_features = result.features
                    features.update(result.features)
                    actual_mode = "online"
                else:
                    fallback_reason = result.reason or "incomplete_online_features"
            if fallback_reason:
                actual_mode = "offline_fallback"
        names = artifact["metadata"]["feature_names"]
        x = np.asarray([[features[name] for name in names]])
        probability = float(artifact["model"].predict_proba(x)[0, 1])
        risk, label = score_probability(probability)
        explainer = self.online_explainer if actual_mode == "online" else self.offline_explainer
        explanation = explainer.explain(url, online_features=online_features)
        return Prediction(
            risk,
            label,
            probability,
            actual_mode,
            artifact["metadata"]["version"],
            explanation.top_features,
            fallback_reason=fallback_reason,
        )

    def predict_email(self, text: str) -> Prediction:
        """Preserve Stage 3 cleaning, input limits, exact terms and cleaned-text highlights."""
        if self.email is None:
            raise ValueError("Email model is not loaded")
        explanation = self.email_explainer.explain(text)
        probability = float(self.email["model"].predict_proba([text])[0, 1])
        risk, label = score_probability(probability)
        return Prediction(
            risk,
            label,
            probability,
            "offline",
            self.email["metadata"]["version"],
            explanation.top_features,
            explanation.highlights,
            explanation.cleaned_text,
        )


def main() -> None:
    """Offer a spawn-safe URL CLI while API implementation remains a separate stage."""
    import argparse
    import json
    from dataclasses import asdict

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    try:
        result = Predictor.from_directory(args.artifacts).predict_url(args.url, args.mode)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Prediction unavailable: {exc}\n")
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
