"""Keep calibrated probability thresholds independent from rounded display scores."""

import math
from typing import Literal

from phishlens.config import PHISHING_CUTOFF, SUSPICIOUS_CUTOFF


def score_probability(
    probability: float,
) -> tuple[int, Literal["legitimate", "suspicious", "phishing"]]:
    """Round to 0–100 for display; choose labels from the unrounded probability."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and between 0 and 1")
    label = "legitimate"
    if probability >= PHISHING_CUTOFF:
        label = "phishing"
    elif probability >= SUSPICIOUS_CUTOFF:
        label = "suspicious"
    return round(100 * probability), label
