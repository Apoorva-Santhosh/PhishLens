"""Check text leakage, exact sparse contributions, safe highlights and registry coexistence."""

import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phishlens.config import MAX_EMAIL_LENGTH
from phishlens.explain.email_words import EmailExplainer
from phishlens.features.email_text import clean_text, text_key
from phishlens.models.prepare_email import SOURCES, prepare
from phishlens.models.registry import load_artifact, save_artifact
from phishlens.models.train_email import load_emails, split_emails, train


@pytest.fixture
def email_artifact(tmp_path: Path) -> Path:
    """Exercise the actual training entry point on separable but varied test documents."""
    rows = [
        {
            "text": (
                "urgent verify password account " if i % 2 else "meeting project agenda schedule "
            )
            + f"case{i}",
            "label": i % 2,
        }
        for i in range(160)
    ]
    path = tmp_path / "emails.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    metadata = train(path, tmp_path / "artifacts")
    assert metadata["test_metrics"]["recall"] > 0.9
    return tmp_path / "artifacts" / "email.joblib"


def test_cleaning() -> None:
    text = "<p>Hello &amp; WORLD</p><script>secret</script><style>hidden</style>"
    assert clean_text(text) == "hello & world"
    cleaned = clean_text("Contact <person@example.com> and https://evil.test/login now")
    assert "emailtoken" in cleaned and "urltoken" in cleaned
    assert "evil.test" not in cleaned and "person@example" not in cleaned
    assert text_key("<b>HELLO</b> world!") == text_key("hello world")


@pytest.mark.parametrize(
    "text",
    ["", "   ", "<script>hidden</script>", "!", "x" * (MAX_EMAIL_LENGTH + 1)],
    ids=["empty", "whitespace", "hidden-html", "punctuation", "oversized"],
)
def test_bad_text(text: str) -> None:
    with pytest.raises(ValueError):
        clean_text(text)


def test_preparation_and_conflicts(tmp_path: Path) -> None:
    for source in SOURCES:
        pd.DataFrame(
            [
                {"subject": "Hello", "body": "world!", "label": 0},
                {"subject": "Verify", "body": "password now", "label": 1},
                {"subject": "Ambiguous", "body": "same text", "label": 0},
                {"subject": "Ambiguous", "body": "same text", "label": 1},
                {"subject": "", "body": "", "label": 0},
            ]
        ).to_csv(tmp_path / f"{source}.csv", index=False)
    report = prepare(tmp_path)
    assert report["rows"] == 2
    assert report["conflicting_normalized_rows_removed"] == 12
    assert report["duplicate_normalized_rows_removed"] == 10
    assert sum(report["rejected"].values()) == 6
    frame = load_emails(tmp_path / "emails.csv")
    assert set(frame.label) == {0, 1}
    assert (tmp_path / "emails.provenance.json").exists()


def test_loader(tmp_path: Path) -> None:
    path = tmp_path / "emails.csv"
    for contents, message in [
        ("wrong,label\nhello,0\n", "columns"),
        ("text,label\nhello,yes\n", "labels"),
        ("text,label\nHello world!,0\nhello world,1\n", "Conflicting"),
        ("text,label\nhello world,0\n", "both classes"),
    ]:
        path.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_emails(path)


def test_splits_and_vocabulary(email_artifact: Path, tmp_path: Path) -> None:
    artifact = load_artifact(email_artifact)
    parts = split_emails(load_emails(tmp_path / "emails.csv"))
    for a, b in combinations(parts.values(), 2):
        assert set(a.key).isdisjoint(b.key)
    repeated = split_emails(load_emails(tmp_path / "emails.csv"))
    for name in parts:
        assert parts[name].index.equals(repeated[name].index)
    names = artifact["metadata"]["feature_names"]
    for text in parts["test"].text:
        assert text.split()[-1] not in names
    assert "verify password" in names


def test_exact_contributions_and_highlights(email_artifact: Path) -> None:
    artifact = load_artifact(email_artifact)
    model = artifact["model"]
    pipeline = model.calibrated_classifiers_[0].estimator.estimator
    explainer = EmailExplainer.from_artifact(email_artifact)
    text = "<b>urgent verify a password</b> meeting project https://example.org"
    result = explainer.explain(text, top_k=100_000)
    assert result == explainer.explain(text, top_k=100_000)
    assert result.explained_value == pytest.approx(pipeline.decision_function([text])[0])
    assert result.base_value + sum(f["raw_contribution"] for f in result.top_features) == (
        pytest.approx(result.explained_value)
    )
    vector = pipeline["tfidf"].transform([text])
    vocabulary = pipeline["tfidf"].vocabulary_
    for feature in result.top_features:
        index = vocabulary[feature["name"]]
        exact = pipeline["classifier"].coef_[0, index] * vector[0, index]
        assert feature["raw_contribution"] == pytest.approx(exact)
        assert feature["contribution"] == pytest.approx(exact * result.calibration_orientation)
    compact = explainer.explain(text, top_k=2)
    assert {f["direction"] for f in compact.top_features} == {"raises_risk", "lowers_risk"}
    assert any(h["term"] == "verify password" for h in result.highlights)
    for highlight in result.highlights:
        assert 0 <= highlight["start"] < highlight["end"] <= len(result.cleaned_text)
    json.dumps(asdict(result), allow_nan=False)
    unknown = explainer.explain("qzxwvbnmlkjhg")
    assert unknown.top_features == unknown.highlights == []
    assert unknown.explained_value == unknown.base_value
    with pytest.raises(ValueError, match="top_k"):
        explainer.explain(text, top_k=1)
    with pytest.raises(ValueError, match="at most"):
        explainer.explain("x" * (MAX_EMAIL_LENGTH + 1))


def test_calibration_and_registry(email_artifact: Path, tmp_path: Path) -> None:
    artifact = load_artifact(email_artifact)
    model = artifact["model"]
    text = "urgent verify password meeting"
    original = EmailExplainer(model).explain(text)
    model.calibrated_classifiers_[0].calibrators[0].a_ *= -1
    reversed_result = EmailExplainer(model).explain(text)
    assert reversed_result.calibration_orientation == -original.calibration_orientation
    original_map = {f["name"]: f["contribution"] for f in original.top_features}
    for feature in reversed_result.top_features:
        assert feature["contribution"] == -original_map[feature["name"]]
    assert reversed_result.explained_value == original.explained_value
    url_path = save_artifact(tmp_path, "url marker", {"version": "test"})
    save_artifact(tmp_path, model, artifact["metadata"], name="email")
    assert load_artifact(url_path)["model"] == "url marker"
    with pytest.raises(ValueError, match="Unknown"):
        save_artifact(tmp_path, model, {}, name="../bad")
    assert np.isfinite(model.predict_proba([text])).all()
