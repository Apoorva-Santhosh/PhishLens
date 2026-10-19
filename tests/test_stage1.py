"""Check leakage boundaries and a real three-model training round trip."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from phishlens.explain.url_shap import URLExplainer
from phishlens.features.url_offline import (
    FEATURE_NAMES,
    entropy,
    extract_features,
    registered_domain,
)
from phishlens.models.audit_url import audit
from phishlens.models.data import load_urls, split_domains
from phishlens.models.registry import load_artifact
from phishlens.models.train_url import train


def write_dataset(path):
    rows = []
    for i in range(160):
        label = i % 2
        for sub in ("www", "mail"):
            url = (
                f"http://{sub}.account-{i}.xyz/login?verify=1"
                if label
                else f"https://{sub}.example{i}.com/about"
            )
            rows.append({"url": url, "label": label})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_features_and_order():
    values = extract_features("https://user@a.example.co.uk:443/login?x=1&y=%20")
    assert tuple(values) == FEATURE_NAMES
    assert values["subdomain_count"] == 1
    assert values["has_port"] == values["uses_https"] == values["at_count"] == 1
    assert registered_domain("https://a.example.co.uk/path") == "example.co.uk"
    assert registered_domain("https://b.example.co.uk") == "example.co.uk"
    assert extract_features("http://127.0.0.1:80")["ip_host"] == 1
    assert extract_features("http://[::1]:80")["ip_host"] == 1
    assert extract_features("https://bÃƒÆ’Ã‚Â¼cher.de")["punycode_host"] == 1
    assert extract_features("https://bÃƒÆ’Ã‚Â¼cher.de")["non_ascii_host"] == 1
    assert extract_features("bit.ly/x")["shortener_host"] == 1
    assert entropy("") == entropy("aaaa") == 0
    assert entropy("ab") == 1
    assert all(np.isfinite(list(values.values())))


@pytest.mark.parametrize(
    "url",
    ["", "https://", "ftp://example.com", "a b.com", "https://example.com:bad", "https://[broken"],
)
def test_bad_urls(url):
    with pytest.raises(ValueError):
        extract_features(url)


def test_loader_validation(tmp_path):
    path = tmp_path / "urls.csv"
    path.write_text("wrong,label\na,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        load_urls(path)
    path.write_text("url,label\nexample.com,good\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Labels"):
        load_urls(path)
    path.write_text("url,label\nexample.com,0\nexample.com,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting"):
        load_urls(path)
    path.write_text("url,label\nexample.com,0\nexample.com,0\nother.com,1\n", encoding="utf-8")
    assert len(load_urls(path)) == 2


def test_no_domain_leakage(tmp_path):
    path = tmp_path / "urls.csv"
    write_dataset(path)
    parts = split_domains(load_urls(path))
    for left, right in combinations(parts.values(), 2):
        assert set(left.domain).isdisjoint(right.domain)
    assert sum(map(len, parts.values())) == 320
    repeated = split_domains(load_urls(path))
    for name in parts:
        assert parts[name].index.equals(repeated[name].index)


def test_training_round_trip(tmp_path):
    path = tmp_path / "urls.csv"
    write_dataset(path)
    metadata = train(path, tmp_path / "models")
    assert len(metadata["validation_metrics"]) == 3
    scores = metadata["validation_metrics"]
    expected = max(scores, key=lambda name: (scores[name]["roc_auc"], scores[name]["recall"]))
    assert metadata["model_type"] == expected
    assert 0 <= metadata["test_metrics"]["roc_auc"] <= 1
    artifact = load_artifact(tmp_path / "models" / "url_offline.joblib")
    x = np.array([list(extract_features("https://example.com").values())])
    probability = artifact["model"].predict_proba(x)[0, 1]
    assert 0 <= probability <= 1
    assert artifact["metadata"]["feature_names"] == list(FEATURE_NAMES)
    assert (tmp_path / "models" / "url_report.md").exists()
    explanation = URLExplainer.from_artifact(tmp_path / "models" / "url_offline.joblib")
    assert len(explanation.explain("https://example.com").top_features) == 8
    assert metadata["explanation"]["calibrated_probability_explained"] is False
    report = audit(
        path,
        tmp_path / "models/url_offline.joblib",
        tmp_path / "models/url_offline.joblib",
        tmp_path / "audit.json",
    )
    assert report["test_domains_disjoint"]
    assert report["sanity_probabilities_finite"]
    assert len(report["sanity_probes"]) == 8


def test_features_never_connect(monkeypatch):
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("Offline extraction attempted network access")

    monkeypatch.setattr(socket, "socket", blocked)
    assert extract_features("https://new.unseen-example.co.uk/login")["subdomain_count"] == 1
    assert registered_domain("https://new.unseen-example.co.uk") == "unseen-example.co.uk"


def test_metric_class_orientation():
    from phishlens.models.evaluate import evaluate

    metrics = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.9]))
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(0.75)
