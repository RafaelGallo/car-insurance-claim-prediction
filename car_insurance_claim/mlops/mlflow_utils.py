from typing import Optional

from loguru import logger

from car_insurance_claim.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

# every call here is best-effort: the Streamlit app and the CLIs must keep
# working even when the MLflow server isn't reachable (e.g. running without
# docker compose), so failures are logged and swallowed, never raised.


def setup_mlflow(tracking_uri: str = MLFLOW_TRACKING_URI, experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> bool:
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        return True
    except Exception as exc:
        logger.warning(f"MLflow tracking server unreachable at {tracking_uri}: {exc}")
        return False


def log_batch_inference(
    model_name: str,
    n_rows: int,
    n_flagged: int,
    threshold: float,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> Optional[str]:
    """Log one batch-scoring event as its own MLflow run, so production usage
    (which model, how many rows, flagged rate) shows up in the MLflow UI
    alongside training runs. Returns the run id, or None if MLflow is down."""
    try:
        import mlflow

        if not setup_mlflow(tracking_uri, experiment_name):
            return None
        with mlflow.start_run(run_name=f"inference_{model_name}") as run:
            mlflow.set_tags({"model_name": model_name, "stage": "batch_inference", "source": "streamlit"})
            mlflow.log_param("threshold", threshold)
            mlflow.log_metrics(
                {
                    "n_rows": n_rows,
                    "n_flagged": n_flagged,
                    "flagged_rate": n_flagged / n_rows if n_rows else 0.0,
                }
            )
        return run.info.run_id
    except Exception as exc:
        logger.warning(f"Could not log inference run to MLflow: {exc}")
        return None


def get_latest_run_metrics(model_name: str, tracking_uri: str = MLFLOW_TRACKING_URI, experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> Optional[dict]:
    """Latest training run's validation metrics for model_name, to show next
    to the model picker in the Streamlit app. None if MLflow or the run isn't
    available."""
    try:
        import mlflow

        if not setup_mlflow(tracking_uri, experiment_name):
            return None
        # picks up both a real retraining run and the one-off historical
        # backfill, so the sidebar has something to show before the Airflow
        # DAG has ever run; a fresh retrain's later start_time wins.
        runs = mlflow.search_runs(
            filter_string=f"tags.model_name = '{model_name}' and tags.stage != 'batch_inference'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            return None
        row = runs.iloc[0]
        return {col.replace("metrics.", ""): row[col] for col in runs.columns if col.startswith("metrics.val_")}
    except Exception as exc:
        logger.warning(f"Could not fetch latest MLflow metrics for '{model_name}': {exc}")
        return None
