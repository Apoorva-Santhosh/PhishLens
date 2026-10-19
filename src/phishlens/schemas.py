"""Bounded HTTP contracts shared with API clients."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from phishlens.config import MAX_EMAIL_LENGTH, MAX_URL_LENGTH
from phishlens.features.email_text import clean_text
from phishlens.features.url_offline import parse_url

Direction = Literal["raises_risk", "lowers_risk", "neutral"]


class URLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)
    mode: Literal["offline", "online"] = "offline"

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        parse_url(value)
        return value


class EmailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=MAX_EMAIL_LENGTH)

    @field_validator("text")
    @classmethod
    def usable_text(cls, value: str) -> str:
        clean_text(value)
        return value


class FeatureContribution(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    name: str
    value: float
    contribution: float
    direction: Direction
    description: str


class Highlight(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    term: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    direction: Direction
    contribution: float


class PredictionResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    risk_score: int = Field(ge=0, le=100)
    label: Literal["legitimate", "suspicious", "phishing"]
    probability: float = Field(ge=0, le=1)
    mode: Literal["offline", "online", "offline_fallback"]
    model_version: str
    top_features: list[FeatureContribution]
    highlights: list[Highlight] = Field(default_factory=list)
    cleaned_text: str | None = None
    fallback_reason: str | None = None


class ModelStatus(BaseModel):
    version: str | None
    available: bool
    experimental: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    models: dict[str, ModelStatus]
