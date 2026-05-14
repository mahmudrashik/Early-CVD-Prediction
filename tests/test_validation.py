"""Tests for configuration validation."""

import pytest
from infrastructure.validation import validate_config, validate_serve_params


def test_valid_config():
    """Test that valid configuration passes validation."""
    config = {
        "project_name": "Test",
        "random_seed": 42,
        "validation": {
            "threshold": 0.5,
            "bootstrap_iterations": 200,
            "thresholds": [0.1, 0.5, 0.9],
        },
        "models": {
            "include": ["logistic_l2", "random_forest"],
        },
    }
    # Should not raise
    validate_config(config)


def test_invalid_random_seed_negative():
    """Test that negative random_seed raises error."""
    config = {
        "random_seed": -5,
        "validation": {},
        "models": {"include": []},
    }
    with pytest.raises(ValueError, match="random_seed"):
        validate_config(config)


def test_invalid_threshold_out_of_range():
    """Test that threshold outside [0, 1] raises error."""
    config = {
        "random_seed": 42,
        "validation": {"threshold": 1.5},
        "models": {"include": []},
    }
    with pytest.raises(ValueError, match="threshold"):
        validate_config(config)


def test_invalid_bootstrap_iterations():
    """Test that non-positive bootstrap_iterations raises error."""
    config = {
        "random_seed": 42,
        "validation": {"bootstrap_iterations": 0},
        "models": {"include": []},
    }
    with pytest.raises(ValueError, match="bootstrap_iterations"):
        validate_config(config)


def test_invalid_thresholds_values():
    """Test that invalid threshold values in list raise error."""
    config = {
        "random_seed": 42,
        "validation": {
            "threshold": 0.5,
            "thresholds": [0.1, 1.5, 0.9],  # 1.5 is invalid
        },
        "models": {"include": []},
    }
    with pytest.raises(ValueError, match="thresholds"):
        validate_config(config)


def test_invalid_models():
    """Test that invalid model names raise error."""
    config = {
        "random_seed": 42,
        "validation": {},
        "models": {
            "include": ["logistic_l2", "invalid_model"],
        },
    }
    with pytest.raises(ValueError, match="Invalid model names"):
        validate_config(config)


def test_valid_serve_params():
    """Test that valid serve parameters pass validation."""
    validate_serve_params("127.0.0.1", 8000)


def test_invalid_port_too_high():
    """Test that port > 65535 raises error."""
    with pytest.raises(ValueError, match="port"):
        validate_serve_params("127.0.0.1", 70000)


def test_invalid_port_too_low():
    """Test that port < 1 raises error."""
    with pytest.raises(ValueError, match="port"):
        validate_serve_params("127.0.0.1", 0)


def test_invalid_port_string():
    """Test that non-integer port raises error."""
    with pytest.raises(ValueError, match="port"):
        validate_serve_params("127.0.0.1", "invalid")  # type: ignore
