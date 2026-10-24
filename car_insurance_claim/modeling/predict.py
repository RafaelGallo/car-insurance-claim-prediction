from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import typer
from loguru import logger

from car_insurance_claim.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

ID_COLUMN = "policy_id"
TARGET_COLUMN = "is_claim"

# display name -> joblib filename under MODELS_DIR
MODEL_REGISTRY = {
    "HistGradientBoosting": "histgradientboosting.joblib",
    "CatBoost": "catboost.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
    "LightGBM": "lightgbm.joblib",
    "XGBoost": "xgboost.joblib",
    "AdaBoost": "adaboost.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Extra Trees": "extra_trees.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Pipeline HGB (Busca Aleatoria)": "pipeline_hgb_busca_aleatoria.joblib",
    "Pipeline HGB (Grid Search)": "pipeline_hgb_grade.joblib",
}

# every model in MODEL_REGISTRY — including the two tuned HGB pipelines,
# whose own ColumnTransformer is a numeric-only imputer — was fit on
# label-encoded columns (see notebooks/model_ml.ipynb and pipeline_ml.ipynb,
# which both encode categoricals in place before splitting X/y). There is no
# raw-input model in this registry.


def list_available_models(models_dir: Path = MODELS_DIR) -> dict:
    """Model display names whose joblib file actually exists on disk."""
    return {name: filename for name, filename in MODEL_REGISTRY.items() if (models_dir / filename).exists()}


def load_label_encoders(models_dir: Path = MODELS_DIR) -> dict:
    return joblib.load(models_dir / "label_encoders.joblib")


def load_feature_columns(models_dir: Path = MODELS_DIR) -> list:
    return joblib.load(models_dir / "colunas.joblib")


def load_business_threshold(models_dir: Path = MODELS_DIR) -> float:
    payload = joblib.load(models_dir / "limiar_negocio.joblib")
    return float(payload["limiar"])


def load_model(model_name: str, models_dir: Path = MODELS_DIR):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    model_path = models_dir / MODEL_REGISTRY[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def validate_feature_columns(df: pd.DataFrame, feature_columns: list) -> None:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def encode_categoricals(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """Apply the fitted LabelEncoders, mapping unseen categories to the
    encoder's first known class instead of raising, since a CSV uploaded in
    production can contain values the training data never saw."""
    df = df.copy()
    for col, encoder in label_encoders.items():
        if col not in df.columns:
            continue
        values = df[col].astype(str)
        known = set(encoder.classes_)
        unseen_mask = ~values.isin(known)
        if unseen_mask.any():
            logger.warning(
                f"Column '{col}' has {int(unseen_mask.sum())} unseen category "
                f"value(s); mapping them to '{encoder.classes_[0]}'."
            )
            values = values.where(~unseen_mask, encoder.classes_[0])
        df[col] = encoder.transform(values)
    return df


def prepare_features(
    df_raw: pd.DataFrame,
    feature_columns: list,
    label_encoders: dict,
) -> pd.DataFrame:
    validate_feature_columns(df_raw, feature_columns)
    df = df_raw[feature_columns].copy()
    return encode_categoricals(df, label_encoders)


def predict(
    df_raw: pd.DataFrame,
    model_name: str,
    models_dir: Path = MODELS_DIR,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Score df_raw with the chosen model and append claim_probability /
    claim_prediction columns, keeping every original column (policy_id
    included) so the caller can trace predictions back to the input rows."""
    feature_columns = load_feature_columns(models_dir)
    label_encoders = load_label_encoders(models_dir)
    model = load_model(model_name, models_dir)

    X = prepare_features(df_raw, feature_columns, label_encoders)

    if threshold is None:
        threshold = load_business_threshold(models_dir)

    proba = model.predict_proba(X)[:, 1]

    result = df_raw.copy()
    result["claim_probability"] = proba
    result["claim_prediction"] = (proba >= threshold).astype(int)
    return result


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


@app.command()
def main(
    input_path: Path = typer.Option(..., help="CSV or parquet file with the raw feature columns."),
    model_name: str = typer.Option("HistGradientBoosting", help="Display name from MODEL_REGISTRY."),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "predictions.csv"),
    threshold: Optional[float] = typer.Option(None, help="Override the business threshold from limiar_negocio.joblib."),
    models_dir: Path = typer.Option(MODELS_DIR),
):
    logger.info(f"Loading input data from {input_path}")
    df_raw = _read_table(input_path)

    logger.info(f"Scoring {len(df_raw)} rows with model '{model_name}'")
    result = predict(df_raw, model_name, models_dir=models_dir, threshold=threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path} ({result['claim_prediction'].sum()} flagged as claim)")


if __name__ == "__main__":
    app()
