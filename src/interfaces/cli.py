from __future__ import annotations

import typer

from application.reporting import write_data_and_leakage_reports
from application.train import train_and_evaluate
from infrastructure.config import load_config
from infrastructure.validation import validate_serve_params

app = typer.Typer(help="Early CVD Prediction research pipeline.")


@app.command()
def generate_report(config: str = typer.Option("configs/default.yaml", "--config", "-c")) -> None:
    """Generate data, leakage, architecture, manuscript, and supplementary report scaffolds."""
    cfg = load_config(config)
    write_data_and_leakage_reports(cfg)
    typer.echo("Generated audit and report files.")


@app.command()
def train(config: str = typer.Option("configs/default.yaml", "--config", "-c")) -> None:
    """Run site-aware model comparison, select the champion, and save the model bundle."""
    cfg = load_config(config)
    result = train_and_evaluate(cfg)
    typer.echo(f"Selected champion: {result['champion']}")
    typer.echo(f"Model bundle: {result['model_bundle']}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the FastAPI localhost application."""
    # Validate parameters
    validate_serve_params(host, port)
    
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()

