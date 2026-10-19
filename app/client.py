"""Small HTTP boundary with safe, actionable user-facing errors."""

import os

import httpx
from pydantic import ValidationError

from phishlens.schemas import HealthResponse, PredictionResponse


class APIError(Exception):
    """An error safe to display without echoing submitted content."""


def api_url() -> str:
    """Read the trusted server-side API destination without accepting browser overrides."""
    return os.environ.get("PHISHLENS_API_URL", "http://127.0.0.1:8000").rstrip("/")


def request(path: str, payload: dict | None = None) -> dict:
    """Never follow redirects or use environment proxies for submitted email text."""
    try:
        with httpx.Client(
            base_url=api_url(),
            timeout=httpx.Timeout(3 if payload is None else 60, connect=3),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            response = client.get(path) if payload is None else client.post(path, json=payload)
    except httpx.TimeoutException:
        raise APIError("The analysis timed out. Try again, or use Offline mode.") from None
    except httpx.HTTPError:
        raise APIError("Cannot reach the API. Start the API server and retry.") from None
    if response.status_code == 429:
        wait = response.headers.get("retry-after", "60")
        wait = wait if wait.isdigit() and len(wait) <= 5 else "60"
        raise APIError(f"Request limit reached. Retry in {wait} seconds.")
    errors = {
        413: "The request is too large. Shorten the submitted content.",
        422: "Check the input. Online mode requires a public domain; empty text is not accepted.",
        503: "The requested model is unavailable. Check the API's local model artifacts.",
    }
    if response.status_code != 200:
        raise APIError(errors.get(response.status_code, "Analysis unavailable. Please retry."))
    try:
        return response.json()
    except ValueError:
        raise APIError("The API returned an unreadable response.") from None


def health() -> HealthResponse:
    """Validate availability metadata before enabling analysis controls."""
    try:
        return HealthResponse.model_validate(request("/health"))
    except ValidationError:
        raise APIError("The API returned an incompatible health response.") from None


def analyze(kind: str, value: str, mode: str = "offline") -> PredictionResponse:
    """Submit one analysis and validate its model identity and explanation contract."""
    if kind not in {"url", "email"}:
        raise ValueError("Unsupported analysis kind")
    payload = {"url": value, "mode": mode} if kind == "url" else {"text": value}
    try:
        return PredictionResponse.model_validate(request(f"/predict/{kind}", payload))
    except ValidationError:
        raise APIError("The API returned an incompatible prediction response.") from None
