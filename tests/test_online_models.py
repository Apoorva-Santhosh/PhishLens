"""Train online artifacts only from synthetic snapshot files; never call live transports."""

import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from phishlens.explain.url_shap import URLExplainer
from phishlens.features import url_online
from phishlens.features.url_offline import FEATURE_NAMES, extract_features
from phishlens.features.url_online import ONLINE_NAMES, OnlineLookup, OnlineResult, OnlineSettings
from phishlens.models.data import split_domains
from phishlens.models.precompute_online import collect
from phishlens.models.registry import load_artifact
from phishlens.models.train_url import train as train_offline
from phishlens.models.train_url_online import load_snapshot, train
from phishlens.predict import Predictor
from phishlens.scoring import score_probability


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("stage4-models")
    rows, snapshots = [], []
    for i in range(160):
        domain = f"account-{i}.net" if i % 2 else f"brand{i}.com"
        url = f"http://{domain}/login?verify=1" if i % 2 else f"https://www.{domain}/about"
        rows.append({"url": url, "label": i % 2})
        values = [5.0 if i % 2 else 10_000.0, 1.0, 1.0, 1.0, 2.0, 60.0, 300.0]
        snapshots.append(
            {
                "domain": domain,
                "status": "ok",
                "observed_at": 1_700_000_000.0,
                **dict(zip(ONLINE_NAMES, values, strict=True)),
            }
        )
    data, snapshot = directory / "urls.csv", directory / "snapshot.csv"
    pd.DataFrame(rows).to_csv(data, index=False)
    pd.DataFrame(snapshots).to_csv(snapshot, index=False)
    # No lookup operation is part of either trainer. Block transports during fitting.
    with pytest.MonkeyPatch.context() as patch:
        forbidden = MagicMock(side_effect=AssertionError("Training attempted network acquisition"))
        patch.setattr(url_online, "query_dns", forbidden)
        patch.setattr(url_online, "query_whois", forbidden)
        patch.setattr(url_online, "run_bounded", forbidden)
        train_offline(data, directory)
        metadata = train(data, snapshot, directory)
        forbidden.assert_not_called()
    return directory, data, snapshot, metadata


def test_training_and_explanation(artifacts) -> None:
    directory, data, snapshot, metadata = artifacts
    assert metadata["rows_with_complete_snapshots"] == 160
    assert len(metadata["validation_metrics"]) == 3
    assert metadata["test_metrics"]["roc_auc"] > 0.9
    assert (directory / "url_offline.joblib").exists()
    assert (directory / "url_online.joblib").exists()
    assert (directory / "url_online_report.md").exists()
    assert metadata["feature_names"] == [*FEATURE_NAMES, *ONLINE_NAMES]
    frame = pd.read_csv(data)
    frame["domain"] = frame.url.map(url_online.safe_domain)
    for a, b in combinations(split_domains(frame).values(), 2):
        assert set(a.domain).isdisjoint(b.domain)
    online = load_artifact(directory / "url_online.joblib")
    explainer = URLExplainer.from_artifact(directory / "url_online.joblib")
    values = load_snapshot(snapshot).iloc[1][list(ONLINE_NAMES)].astype(float).to_dict()
    url = "http://account-1.net/login?verify=1"
    result = explainer.explain(url, top_k=30, online_features=values)
    assert result == explainer.explain(url, top_k=30, online_features=values)
    assert len(result.top_features) == 30
    assert any(f["name"] == "domain_age_days" for f in result.top_features)
    base = online["model"].calibrated_classifiers_[0].estimator.estimator
    x = np.array([[*extract_features(url).values(), *(values[name] for name in ONLINE_NAMES)]])
    if metadata["model_type"] == "logistic_regression":
        expected = base.decision_function(x)[0]
    elif metadata["model_type"] == "random_forest":
        expected = base.predict_proba(x)[0, 1]
    else:
        expected = base.predict(x, output_margin=True)[0]
    assert result.explained_value == pytest.approx(
        float(expected) * result.calibration_orientation, abs=1e-5
    )
    with pytest.raises(ValueError, match="complete"):
        explainer.explain(url)
    values["domain_age_days"] = None
    with pytest.raises(ValueError, match="Missing"):
        explainer.explain(url, online_features=values)


def test_prediction_success_and_failure_use_correct_models(artifacts) -> None:
    directory, _, snapshot, _ = artifacts
    offline = load_artifact(directory / "url_offline.joblib")
    online = load_artifact(directory / "url_online.joblib")
    values = load_snapshot(snapshot).iloc[1][list(ONLINE_NAMES)].astype(float).to_dict()
    lookup = MagicMock()
    lookup.lookup.return_value = OnlineResult("account-1.net", values, "ok")
    predictor = Predictor(offline, online, lookup=lookup)
    url = "http://account-1.net/login?verify=1"
    result = predictor.predict_url(url, "online")
    assert result.mode == "online" and result.fallback_reason is None
    assert len(result.top_features) == 8
    lookup.lookup.assert_called_once_with(url)
    lookup.lookup.return_value = OnlineResult("account-1.net", reason="lookup_timeout")
    fallback = predictor.predict_url(url, "online")
    direct = predictor.predict_url(url)
    assert fallback.mode == "offline_fallback" and direct.mode == "offline"
    assert fallback.probability == direct.probability
    assert fallback.top_features == direct.top_features
    assert fallback.fallback_reason == "lookup_timeout"
    assert fallback.risk_score == score_probability(fallback.probability)[0]
    json.dumps(asdict(result), allow_nan=False)


def test_unavailable_model_and_unsafe_input(artifacts) -> None:
    directory, _, _, _ = artifacts
    lookup = MagicMock(side_effect=AssertionError("No network without an online model"))
    predictor = Predictor(load_artifact(directory / "url_offline.joblib"), lookup=lookup)
    assert predictor.predict_url("https://brand0.com/about", "online").mode == "offline_fallback"
    lookup.lookup.assert_not_called()
    with pytest.raises(ValueError):
        predictor.predict_url("http://127.0.0.1/", "online")
    assert predictor.predict_url("http://127.0.0.1/").mode == "offline"
    with pytest.raises(ValueError, match="mode"):
        predictor.predict_url("https://brand0.com", "unknown")
    with pytest.raises(ValueError, match="not loaded"):
        predictor.predict_email("hello world")


def test_snapshot_rejects_malformed_successes(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.csv"
    path.write_text("domain,status\ncustomer.com,ok\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        load_snapshot(path)
    row = {
        "domain": "customer.com",
        "status": "failed",
        "observed_at": 10,
        **dict.fromkeys(ONLINE_NAMES, ""),
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="No complete"):
        load_snapshot(path)
    row.update(status="ok", **dict.fromkeys(ONLINE_NAMES, 1))
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one row"):
        load_snapshot(path)
    row["dns_a_count"] = float("inf")
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        load_snapshot(path)


def test_precomputation_resumes_after_interruption(tmp_path: Path, monkeypatch) -> None:
    data, output = tmp_path / "urls.csv", tmp_path / "snapshot.csv"
    pd.DataFrame([{"url": f"https://brand{i}.com/path", "label": i % 2} for i in range(8)]).to_csv(
        data, index=False
    )
    lookup = OnlineLookup(OnlineSettings(cache_path=tmp_path / "cache.sqlite3", min_interval=0))
    calls = []

    def interrupted(domain, settings):
        calls.append(domain)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return OnlineResult(
            domain,
            dict(zip(ONLINE_NAMES, [1000.0, 1.0, 1.0, 1.0, 2.0, 60.0, 300.0], strict=True)),
            "ok",
        )

    monkeypatch.setattr(url_online, "run_bounded", interrupted)
    with pytest.raises(KeyboardInterrupt):
        collect(data, output, lookup, max_domains=8)
    assert len(pd.read_csv(output)) == 1
    monkeypatch.setattr(
        url_online,
        "run_bounded",
        lambda d, s: OnlineResult(
            d, dict(zip(ONLINE_NAMES, [1000.0, 1.0, 1.0, 1.0, 2.0, 60.0, 300.0], strict=True)), "ok"
        ),
    )
    report = collect(data, output, lookup, max_domains=8)
    assert report["counts"]["resumed"] == 1
    assert report["counts"]["ok"] == 7
    assert len(load_snapshot(output)) == 8
    output.unlink()  # The SQLite cache, not this derived export, is authoritative.
    forbidden = MagicMock(side_effect=AssertionError("Completed domains should not be recollected"))
    monkeypatch.setattr(url_online, "run_bounded", forbidden)
    resumed = collect(data, output, lookup, max_domains=8)
    assert resumed["counts"]["resumed"] == 8
    assert len(load_snapshot(output)) == 8
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "probability,score,label",
    [
        (0.0, 0, "legitimate"),
        (0.3499, 35, "legitimate"),
        (0.35, 35, "suspicious"),
        (0.6999, 70, "suspicious"),
        (0.7, 70, "phishing"),
        (1.0, 100, "phishing"),
    ],
)
def test_scoring_boundaries(probability, score, label) -> None:
    assert score_probability(probability) == (score, label)


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probability(probability) -> None:
    with pytest.raises(ValueError):
        score_probability(probability)
