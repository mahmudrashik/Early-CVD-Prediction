from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p)


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)

