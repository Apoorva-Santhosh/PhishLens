"""Dashboard behavior at the HTTP and Streamlit boundaries."""

from html import unescape
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app import client
from app.visuals import feature_chart, highlighted_text, risk_chart
from phishlens.schemas import HealthResponse, PredictionResponse


@pytest.fixture
def prediction():
    return PredictionResponse(
        risk_score=80,
        label="phishing",
        probability=0.8,
        mode="offline",
        model_version="test",
        top_features=[
            {
                "name": "verify",
                "value": 0.5,
                "contribution": 0.4,
                "direction": "raises_risk",
                "description": "Verification language",
            }
        ],
        cleaned_text="verify account <script>alert(1)</script>",
        highlights=[
            {
                "term": "verify account",
                "start": 0,
                "end": 14,
                "direction": "raises_risk",
                "contribution": 0.4,
            },
            {
                "term": "account",
                "start": 7,
                "end": 14,
                "direction": "lowers_risk",
                "contribution": -0.2,
            },
        ],
    )


def test_highlights_escape_and_preserve_overlapping_text(prediction):
    import re

    markup = highlighted_text(prediction)
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert 'class="risk-up"' in markup
    assert unescape(re.sub(r"<[^>]*>", "", markup)) == prediction.cleaned_text


def test_charts_display_real_risk_and_contributions(prediction):
    assert risk_chart(prediction).layout.annotations[0].text == "80"
    assert risk_chart(None).layout.annotations[0].text == "—"
    assert list(feature_chart(prediction).data[0].x) == [0.4]


@pytest.mark.parametrize(
    "status,message",
    [
        (413, "too large"),
        (422, "Check the input"),
        (429, "Retry in 7"),
        (503, "unavailable"),
        (500, "unavailable"),
    ],
)
def test_http_failures_are_safe(monkeypatch, status, message):
    response = httpx.Response(status, text="private raw content", headers={"retry-after": "7"})
    transport = Mock()
    transport.__enter__ = Mock(return_value=transport)
    transport.__exit__ = Mock(return_value=False)
    transport.post.return_value = response
    monkeypatch.setattr(client.httpx, "Client", Mock(return_value=transport))
    with pytest.raises(client.APIError, match=message) as error:
        client.analyze("email", "secret input")
    assert "private" not in str(error.value)


def test_http_payload_and_schema(monkeypatch, prediction):
    transport = Mock()
    transport.__enter__ = Mock(return_value=transport)
    transport.__exit__ = Mock(return_value=False)
    transport.post.return_value = httpx.Response(200, json=prediction.model_dump())
    monkeypatch.setattr(client.httpx, "Client", Mock(return_value=transport))
    assert client.analyze("url", "python.org", "online") == prediction
    transport.post.assert_called_once_with(
        "/predict/url", json={"url": "python.org", "mode": "online"}
    )
    transport.post.return_value = httpx.Response(200, json={"bad": "shape"})
    with pytest.raises(client.APIError, match="incompatible"):
        client.analyze("email", "hello")


def test_network_timeout(monkeypatch):
    transport = Mock()
    transport.__enter__ = Mock(return_value=transport)
    transport.__exit__ = Mock(return_value=False)
    transport.post.side_effect = httpx.ReadTimeout("private")
    monkeypatch.setattr(client.httpx, "Client", Mock(return_value=transport))
    with pytest.raises(client.APIError, match="timed out"):
        client.analyze("email", "hello")


@pytest.fixture
def connected(monkeypatch):
    health = HealthResponse(
        status="ok",
        models={
            "url_offline": {"version": "test", "available": True},
            "email": {"version": "test", "available": True},
        },
    )
    monkeypatch.setattr(client, "health", lambda: health)


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_streamlit_empty_submit_success_clear_and_error(connected, monkeypatch, prediction):
    analyze = Mock(return_value=prediction)
    monkeypatch.setattr(client, "analyze", analyze)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"),
        default_timeout=15,
    ).run()
    assert not app.exception
    assert len(app.tabs) == 2
    button(app, "Analyze URL").click().run()
    assert app.warning
    analyze.assert_not_called()
    app.text_input[0].set_value("https://python.org")
    button(app, "Analyze URL").click().run()
    assert not app.exception
    assert app.session_state["url_result"] == prediction
    analyze.assert_called_once_with("url", "https://python.org", "offline")
    app.button(key="clear_url").click().run()
    assert "url_result" not in app.session_state
    analyze.side_effect = client.APIError("Cannot reach the API")
    button(app, "Analyze URL").click().run()
    assert app.error
    assert "url_result" not in app.session_state


def test_streamlit_email_and_offline_api(connected, monkeypatch, prediction):
    monkeypatch.setattr(client, "analyze", Mock(return_value=prediction))
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"),
        default_timeout=15,
    ).run()
    app.text_area[0].set_value("verify account")
    button(app, "Analyze email").click().run()
    assert not app.exception
    assert app.session_state["email_result"] == prediction
    monkeypatch.setattr(client, "health", Mock(side_effect=client.APIError("offline")))
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"),
        default_timeout=15,
    ).run()
    assert not app.exception
    assert button(app, "Analyze URL").disabled
    assert button(app, "Analyze email").disabled
    assert app.info
