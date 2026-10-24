from pathlib import Path

import joblib
import pandas as pd
import typer
from loguru import logger

from car_insurance_claim.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODELS_DIR, PROJ_ROOT
from car_insurance_claim.modeling.predict import MODEL_REGISTRY

app = typer.Typer()


def _backfill_table(
    csv_path: Path,
    stage: str,
    eval_set: str,
    models_dir: Path,
    tracking_uri: str,
    experiment_name: str,
) -> int:
    """eval_set is the prefix of the logged metrics: the tree table is measured
    on an inner validation split ("val"), the GLM table on the held-out test
    set ("test"). Columns already named cv_* keep their name."""
    import mlflow
    import mlflow.sklearn

    if not csv_path.exists():
        logger.warning(f"{csv_path} not found; skipping.")
        return 0

    tabela = pd.read_csv(csv_path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logged = 0
    for _, row in tabela.iterrows():
        name = row["modelo"]
        file_name = MODEL_REGISTRY.get(name)
        model_path = models_dir / file_name if file_name else None

        metrics = {k: float(v) for k, v in row.items() if k != "modelo" and pd.notna(v)}

        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"model_name": name, "stage": stage, "eval_set": eval_set})
            mlflow.log_metrics({k if k.startswith("cv_") else f"{eval_set}_{k}": v for k, v in metrics.items()})
            if model_path and model_path.exists():
                model = joblib.load(model_path)
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=name.replace(" ", "_"),
                    serialization_format="pickle",
                )
            else:
                logger.warning(f"No joblib file for '{name}'; logged metrics only, no model artifact.")
        logger.success(f"Backfilled '{name}' ({stage})")
        logged += 1
    return logged


@app.command()
def main(
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_NAME),
    models_dir: Path = typer.Option(MODELS_DIR),
    only_glm: bool = typer.Option(False, help="Log only the GLM table, e.g. after re-running notebooks/glm_poisson.ipynb."),
):
    """One-off backfill: log the already-trained models (the 10-model
    lineup and the 5 GLM models, if their result tables exist) into MLflow,
    so the UI already has history before any retraining DAG has run."""
    output_dir = PROJ_ROOT / "output"
    total = 0
    if not only_glm:
        total += _backfill_table(output_dir / "resultados_modelos.csv", "backfill_ml_models", "val", models_dir, tracking_uri, experiment_name)
    total += _backfill_table(output_dir / "resultados_modelos_glm.csv", "backfill_glm_models", "test", models_dir, tracking_uri, experiment_name)

    if total == 0:
        logger.error("Nothing backfilled — no result tables found under output/.")
        raise typer.Exit(1)

    logger.success(f"Backfilled {total} run(s). Open {tracking_uri} to see them.")


if __name__ == "__main__":
    app()
