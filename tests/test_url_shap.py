"""Verify actual SHAP additivity, class orientation and calibrated artifact integration."""

import json
import socket
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from phishlens.explain.url_shap import URLExplainer
from phishlens.features.url_offline import FEATURE_NAMES, extract_features
from phishlens.models.registry import save_artifact

BENIGN = "https://www.example500.com/about"
PHISHING = "http://account-500.xyz/login?verify=1"


@pytest.fixture(params=["linear", "forest", "xgboost"])
def fitted_model(request: pytest.FixtureRequest) -> CalibratedClassifierCV:
    """Fit every supported estimator and calibrate on separate synthetic domains."""
    urls = [
        f"http://account-{i}.xyz/login?verify=1" if i % 2 else f"https://www.example{i}.com/about"
        for i in range(120)
    ]
    x = np.array([list(extract_features(url).values()) for url in urls])
    y = np.arange(120) % 2
    candidates = {
        "linear": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
        "forest": RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1),
        "xgboost": XGBClassifier(
            n_estimators=20,
            max_depth=2,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            device="cpu",
        ),
    }
    estimator = candidates[request.param].fit(x[:80], y[:80])
    model = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    return model.fit(x[80:], y[80:])


@pytest.mark.parametrize("url", [BENIGN, PHISHING])
def test_additivity_and_risk_direction(fitted_model: CalibratedClassifierCV, url: str) -> None:
    """Reconstruct class-1 output and keep signs fixed even for legitimate predictions."""
    explainer = URLExplainer(fitted_model)
    result = explainer.explain(url, top_k=len(FEATURE_NAMES))
    x = np.array([list(extract_features(url).values())])
    estimator = fitted_model.calibrated_classifiers_[0].estimator.estimator
    if isinstance(estimator, RandomForestClassifier):
        expected = estimator.predict_proba(x)[0, 1]
        assert result.output_space == "base_model_probability"
    elif isinstance(estimator, XGBClassifier):
        expected = estimator.predict(x, output_margin=True)[0]
        assert result.output_space == "base_model_log_odds"
    else:
        expected = estimator.decision_function(x)[0]
        assert result.output_space == "base_model_log_odds"
        scaled = estimator[0].transform(x)[0]
        exact = dict(zip(FEATURE_NAMES, estimator[1].coef_[0] * scaled, strict=True))
        for feature in result.top_features:
            assert feature["contribution"] == pytest.approx(exact[feature["name"]], abs=1e-6)
    assert result.calibration_orientation == 1
    assert result.explained_value == pytest.approx(float(expected), abs=1e-5)
    total = sum(f["contribution"] for f in result.top_features)
    assert result.base_value + total == pytest.approx(float(expected), abs=1e-5)
    probability = fitted_model.predict_proba(x)[0, 1]
    assert (probability > 0.5) == (url == PHISHING)
    assert (total > 0) == (url == PHISHING)
    for feature in result.top_features:
        expected_direction = (
            "raises_risk"
            if feature["contribution"] > 0
            else "lowers_risk"
            if feature["contribution"] < 0
            else "neutral"
        )
        assert feature["direction"] == expected_direction
        assert feature["value"] == extract_features(url)[feature["name"]]
        assert feature["description"]
    json.dumps(asdict(result), allow_nan=False)


def test_deterministic_top_features(fitted_model: CalibratedClassifierCV) -> None:
    """Repeated calls and fresh explainers preserve exact values and tie order."""
    explainer = URLExplainer(fitted_model)
    result = explainer.explain(PHISHING)
    assert result == explainer.explain(PHISHING)
    assert result == URLExplainer(fitted_model).explain(PHISHING)
    assert len(result.top_features) == 8
    assert len(explainer.explain(PHISHING, top_k=5).top_features) == 5
    magnitudes = [abs(f["contribution"]) for f in result.top_features]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_reversed_and_constant_calibration(fitted_model: CalibratedClassifierCV) -> None:
    """A decreasing sigmoid reverses risk signs; a constant sigmoid has no risk influence."""
    original = URLExplainer(fitted_model).explain(PHISHING)
    sigmoid = fitted_model.calibrated_classifiers_[0].calibrators[0]
    sigmoid.a_ *= -1
    reversed_result = URLExplainer(fitted_model).explain(PHISHING)
    assert reversed_result.calibration_orientation == -1
    assert reversed_result.output_space.startswith("oriented_")
    assert reversed_result.explained_value == pytest.approx(-original.explained_value)
    for positive, negative in zip(original.top_features, reversed_result.top_features, strict=True):
        assert positive["name"] == negative["name"]
        assert positive["contribution"] == -negative["contribution"]
        reversed_direction = {
            "raises_risk": "lowers_risk",
            "lowers_risk": "raises_risk",
            "neutral": "neutral",
        }
        assert negative["direction"] == reversed_direction[positive["direction"]]
    sigmoid.a_ = 0
    constant = URLExplainer(fitted_model).explain(PHISHING)
    assert constant.calibration_orientation == 0
    assert all(f["direction"] == "neutral" for f in constant.top_features)
    assert all(f["contribution"] == 0 for f in constant.top_features)


def test_artifact_and_no_network(
    fitted_model: CalibratedClassifierCV, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three calibrated model types survive registry round trips without lookup calls."""
    path = save_artifact(tmp_path, fitted_model, {"feature_names": list(FEATURE_NAMES)})

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Explaining a URL attempted a network connection")

    monkeypatch.setattr(socket, "socket", blocked)
    loaded = URLExplainer.from_artifact(path).explain(PHISHING)
    assert loaded == URLExplainer(fitted_model).explain(PHISHING)
    save_artifact(tmp_path, fitted_model, {"feature_names": list(reversed(FEATURE_NAMES))})
    with pytest.raises(ValueError, match="schema"):
        URLExplainer.from_artifact(path)


def test_invalid_requests(fitted_model: CalibratedClassifierCV) -> None:
    """Fail clearly before presenting misleading explanations."""
    explainer = URLExplainer(fitted_model)
    for top_k in (0, -1, 24, True, 1.5):
        with pytest.raises(ValueError, match="top_k"):
            explainer.explain(PHISHING, top_k=top_k)
    with pytest.raises(ValueError):
        explainer.explain("https://")
    with pytest.raises(ValueError, match="fitted"):
        URLExplainer(CalibratedClassifierCV())
    fitted_model.method = "isotonic"
    with pytest.raises(ValueError, match="sigmoid"):
        URLExplainer(fitted_model)
