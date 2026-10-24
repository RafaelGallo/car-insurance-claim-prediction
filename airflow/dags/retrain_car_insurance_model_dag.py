from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.decorators import dag, task

DATA_RAW_DIR = Path("/opt/airflow/data/raw")
SEED_DATASET = Path("/opt/airflow/data/train.parquet")
MODELS_DIR = Path("/opt/airflow/models")
MLFLOW_TRACKING_URI = "http://mlflow:5000"
BUSINESS_THRESHOLD = 0.55


@dag(
    dag_id="retrain_car_insurance_model",
    description="Batch-retrains the 10-model lineup on the newest file in data/raw (falling back to the seed dataset) and logs every run to MLflow.",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["car-insurance-claim", "mlops", "retraining"],
)
def retrain_car_insurance_model():
    @task
    def find_batch() -> str:
        from car_insurance_claim.modeling.train import find_latest_batch

        path = find_latest_batch(DATA_RAW_DIR, SEED_DATASET)
        logging.getLogger("airflow.task").info(f"Training batch: {path}")
        return str(path)

    @task
    def retrain(data_path: str) -> dict:
        from car_insurance_claim.modeling.train import train_and_log

        tabela = train_and_log(
            Path(data_path),
            models_out_dir=MODELS_DIR,
            threshold=BUSINESS_THRESHOLD,
            tracking_uri=MLFLOW_TRACKING_URI,
        )
        melhor = tabela.iloc[0]
        return {"melhor_modelo": str(melhor["modelo"]), "val_pr_auc": float(melhor["val_pr_auc"])}

    @task
    def report(resultado: dict) -> None:
        logging.getLogger("airflow.task").info(
            f"Retraining complete. Best model: {resultado['melhor_modelo']} "
            f"(val PR-AUC={resultado['val_pr_auc']:.4f}). Artifacts written to {MODELS_DIR}, "
            f"runs logged to {MLFLOW_TRACKING_URI}."
        )

    report(retrain(find_batch()))


retrain_car_insurance_model()
