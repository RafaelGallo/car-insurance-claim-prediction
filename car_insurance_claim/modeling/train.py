from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import typer
from loguru import logger
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from car_insurance_claim.config import (
    DATA_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

ID_COLUMN = "policy_id"
TARGET_COLUMN = "is_claim"
DEFAULT_SEED_DATASET = DATA_DIR / "train.parquet"


def build_models(positive_weight: float) -> dict:
    """The same 10-model lineup trained in notebooks/model_ml.ipynb, with the
    same hyperparameters, so retraining stays comparable to the original run."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=30, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=30, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=positive_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=4, auto_class_weights="Balanced", random_seed=42, verbose=0
        ),
    }


def find_latest_batch(raw_data_dir: Path, fallback_path: Path) -> Path:
    """Most recently modified csv/parquet under raw_data_dir, falling back to
    the original seed dataset when no new batch has been dropped there yet."""
    candidates = sorted(
        (p for p in raw_data_dir.glob("*") if p.suffix.lower() in {".csv", ".parquet"}),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    logger.warning(f"No batch found in {raw_data_dir}; falling back to seed dataset {fallback_path}")
    return fallback_path


def load_training_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def encode_training_data(df: pd.DataFrame):
    """Fit one LabelEncoder per categorical column (+ 'make', which is a
    numeric brand code treated as categorical), mirroring Part 6 of
    notebooks/model_ml.ipynb exactly."""
    df = df.copy()
    cat_cols = df.select_dtypes(exclude="number").columns.drop(ID_COLUMN).tolist() + ["make"]
    encoders = {}
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
    return df, encoders


def split_train_val_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_tr, X_val, X_test, y_tr, y_val, y_test


def evaluate(model, X, y, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, proba),
    }


def _loggable_params(model) -> dict:
    """Only scalar hyperparameters survive — a Pipeline's get_params() also
    returns nested estimator objects, which MLflow can't log as params."""
    if not hasattr(model, "get_params"):
        return {}
    return {k: v for k, v in model.get_params().items() if v is None or isinstance(v, (int, float, str, bool))}


def train_and_log(
    data_path: Path,
    models_out_dir: Path = MODELS_DIR,
    threshold: float = 0.55,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> pd.DataFrame:
    """Retrain the 10-model lineup from scratch on data_path, log every run to
    MLflow (params, metrics and the model artifact, registered by name),
    overwrite the joblib artifacts under models_out_dir, and return the
    ranked comparison table. Called by the Airflow retraining DAG and by the
    `train` CLI below."""
    import mlflow
    import mlflow.sklearn

    logger.info(f"Loading training data from {data_path}")
    df = load_training_data(data_path)

    logger.info("Label-encoding categorical columns")
    df, encoders = encode_training_data(df)

    X = df.drop([TARGET_COLUMN, ID_COLUMN], axis=1)
    y = df[TARGET_COLUMN]
    feature_columns = list(X.columns)

    X_tr, X_val, X_test, y_tr, y_val, y_test = split_train_val_test(X, y)
    positive_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    models = build_models(positive_weight)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    models_out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in models.items():
        logger.info(f"Training {name}")
        model.fit(X_tr, y_tr)
        val_metrics = evaluate(model, X_val, y_val, threshold)
        test_metrics = evaluate(model, X_test, y_test, threshold)

        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"model_name": name, "stage": "batch_retrain", "source_file": data_path.name})
            mlflow.log_param("threshold", threshold)
            mlflow.log_params(_loggable_params(model))
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=name.replace(" ", "_"),
                serialization_format="pickle",
            )

        file_name = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, models_out_dir / file_name)

        # unprefixed metric names, so this CSV's "roc_auc"/"pr_auc" columns
        # stay compatible with the original notebook export and with the
        # Streamlit comparison tab, across every future retraining run.
        rows.append({"modelo": name, **val_metrics})
        logger.success(f"{name}: val PR-AUC = {val_metrics['pr_auc']:.4f}")

    joblib.dump(encoders, models_out_dir / "label_encoders.joblib")
    joblib.dump(feature_columns, models_out_dir / "colunas.joblib")
    joblib.dump({"limiar": threshold}, models_out_dir / "limiar_negocio.joblib")

    tabela = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    # sibling of models_out_dir by default (PROJ_ROOT/output for the real
    # models/ dir), so pointing models_out_dir elsewhere for a test run never
    # clobbers the project's real result table.
    output_dir = models_out_dir.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(output_dir / "resultados_modelos.csv", index=False)
    return tabela


@app.command()
def main(
    data_path: Optional[Path] = typer.Option(
        None, help="CSV/parquet batch to train on; defaults to the newest file in data/raw, falling back to data/train.parquet."
    ),
    threshold: float = typer.Option(0.55, help="Business decision threshold saved to limiar_negocio.joblib."),
    models_dir: Path = typer.Option(MODELS_DIR),
):
    resolved_path = data_path or find_latest_batch(RAW_DATA_DIR, DEFAULT_SEED_DATASET)
    tabela = train_and_log(resolved_path, models_out_dir=models_dir, threshold=threshold)
    logger.success(f"Retraining complete. Ranked results:\n{tabela.round(4).to_string()}")


if __name__ == "__main__":
    app()
