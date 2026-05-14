from __future__ import annotations


def risk_category(probability: float) -> str:
    """Return a cautious research risk category from a calibrated probability."""
    if probability < 0.20:
        return "low"
    if probability < 0.50:
        return "moderate"
    return "elevated"


def risk_message(probability: float) -> str:
    category = risk_category(probability)
    if category == "low":
        return "The model estimates a lower probability of recorded heart disease for this benchmark profile."
    if category == "moderate":
        return "The model estimates an intermediate probability of recorded heart disease for this benchmark profile."
    return "The model estimates an elevated probability of recorded heart disease for this benchmark profile."

