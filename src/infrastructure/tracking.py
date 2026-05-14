from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def mlflow_run(experiment_name: str, run_name: str, tracking_dir: str | Path) -> Iterator[object | None]:
    try:
        import mlflow

        tracking_uri = Path(tracking_dir).resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except Exception:
        yield None


def log_metrics(metrics: dict[str, float]) -> None:
    try:
        import mlflow

        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
    except Exception:
        return

