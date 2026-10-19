"""Share bounded, deterministic text cleaning between fitting and inference."""

import hashlib
import re
from html.parser import HTMLParser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishlens.config import EMAIL_MAX_FEATURES, MAX_EMAIL_LENGTH, SEED

URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        if not self.hidden:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        if not self.hidden:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def clean_text(text: str) -> str:
    """Remove markup and identity strings without fetching links or evaluating HTML."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Email text must be a nonempty string")
    if len(text) > MAX_EMAIL_LENGTH:
        raise ValueError(f"Email text must be at most {MAX_EMAIL_LENGTH} characters")
    # Normalize before parsing so plain-text addresses like <user@example.com> survive.
    text = URL_PATTERN.sub(" urltoken ", text)
    text = EMAIL_PATTERN.sub(" emailtoken ", text)
    parser = _VisibleText()
    parser.feed(text)
    parser.close()
    cleaned = " ".join(" ".join(parser.parts).lower().split())
    if not re.search(r"\b\w\w+\b", cleaned):
        raise ValueError("Email contains no usable text after cleaning")
    return cleaned


def text_key(text: str) -> str:
    """Group identical TF-IDF token sequences despite markup and punctuation differences."""
    tokens = re.findall(r"(?u)\b\w\w+\b", clean_text(text))
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()


def build_pipeline() -> Pipeline:
    """Fit vocabulary and IDF only inside the training pipeline to avoid leakage."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    lowercase=False,
                    ngram_range=(1, 2),
                    max_features=EMAIL_MAX_FEATURES,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=SEED, solver="liblinear"),
            ),
        ]
    )
