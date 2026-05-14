"""Parameter validation for project configuration."""

from __future__ import annotations

import warnings
from typing import Any


VALID_MODELS = {
    "logistic_l2",
    "random_forest",
    "gradient_boosting",
    "calibrated_gradient_boosting",
    "mlp_notebook_baseline",
}


class ValidationWarning(UserWarning):
    """Warning for unusual but not necessarily invalid configuration values."""


def validate_config(config: dict[str, Any]) -> None:
    """Validate core project configuration values."""
    errors: list[str] = []

    try:
        seed = int(config.get("random_seed", 42))
        if seed < 0:
            errors.append("random_seed must be non-negative.")
    except (ValueError, TypeError):
        errors.append("random_seed must be an integer.")

    validation = config.get("validation", {})

    try:
        threshold = float(validation.get("threshold", 0.5))
        if not 0 <= threshold <= 1:
            errors.append("validation.threshold must be between 0 and 1.")
    except (ValueError, TypeError):
        errors.append("validation.threshold must be a float between 0 and 1.")

    try:
        bootstrap_iterations = int(validation.get("bootstrap_iterations", 200))
        if bootstrap_iterations <= 0:
            errors.append("validation.bootstrap_iterations must be positive.")
    except (ValueError, TypeError):
        errors.append("validation.bootstrap_iterations must be a positive integer.")

    thresholds = validation.get("thresholds", [0.5])
    if not isinstance(thresholds, list):
        errors.append("validation.thresholds must be a list.")
    else:
        for index, value in enumerate(thresholds):
            try:
                threshold_value = float(value)
            except (ValueError, TypeError):
                errors.append(f"validation.thresholds[{index}] must be numeric.")
                continue
            if not 0 <= threshold_value <= 1:
                errors.append(f"validation.thresholds[{index}] must be between 0 and 1.")

    included_models = config.get("models", {}).get("include", [])
    if not isinstance(included_models, list):
        errors.append("models.include must be a list.")
    else:
        invalid_models = set(included_models) - VALID_MODELS
        if invalid_models:
            errors.append(f"Invalid model names in models.include: {sorted(invalid_models)}.")

    if errors:
        raise ValueError("Configuration validation failed:\n\n" + "\n".join(errors))


def validate_serve_params(host: str, port: int) -> None:
    """Validate host and port values for the serve command."""
    errors: list[str] = []

    try:
        port_value = int(port)
    except (ValueError, TypeError):
        errors.append("port must be an integer.")
    else:
        if not 1 <= port_value <= 65535:
            errors.append("port must be between 1 and 65535.")

    if not isinstance(host, str) or not host.strip():
        errors.append("host must be a non-empty string.")

    if errors:
        raise ValueError("Serve parameters validation failed:\n\n" + "\n".join(errors))


def warn_about_parameter(param_name: str, current_value: Any, valid_range: str) -> None:
    """Warn about an unusual parameter value."""
    message = (
        f"Parameter '{param_name}' may be unusual.\n"
        f"Current value: {current_value}\n"
        f"Valid range/options: {valid_range}"
    )
    warnings.warn(message, ValidationWarning, stacklevel=3)
