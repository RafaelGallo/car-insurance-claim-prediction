import os

import pandas as pd
import streamlit as st

from car_insurance_claim.config import MODELS_DIR, PROJ_ROOT
from car_insurance_claim.mlops.mlflow_utils import get_latest_run_metrics, log_batch_inference
from car_insurance_claim.modeling.predict import (
    list_available_models,
    load_business_threshold,
    load_feature_columns,
    predict,
)

MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000")

st.set_page_config(page_title="Car Insurance Claim — Risk Scoring", page_icon="🚗", layout="wide")


@st.cache_data
def _cached_feature_columns():
    return load_feature_columns()


@st.cache_data
def _cached_threshold():
    return load_business_threshold()


@st.cache_data
def _cached_available_models():
    return list_available_models()


@st.cache_data
def _read_uploaded(file) -> pd.DataFrame:
    if file.name.lower().endswith(".parquet"):
        return pd.read_parquet(file)
    return pd.read_csv(file)


@st.cache_data
def _load_results_table(path):
    if path.exists():
        return pd.read_csv(path)
    return None


st.title("🚗 Car Insurance Claim — Risk Scoring")
st.caption("Previsão em lote de risco de sinistro, com seleção de modelo e métricas no MLflow.")

available_models = _cached_available_models()
feature_columns = _cached_feature_columns()
default_threshold = _cached_threshold()

with st.sidebar:
    st.header("Configuração")
    model_name = st.selectbox("Modelo", options=list(available_models.keys()), index=0)
    threshold = st.slider(
        "Limiar de decisão (probabilidade ≥ limiar ⇒ sinistro)",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01,
        help=f"Padrão carregado de models/limiar_negocio.joblib ({default_threshold:.2f}).",
    )
    log_to_mlflow = st.checkbox("Registrar esta execução no MLflow", value=True)

    st.divider()
    st.markdown(f"**Modelo selecionado:** `{model_name}`")
    metrics = get_latest_run_metrics(model_name)
    if metrics:
        st.markdown("**Última métrica de validação (MLflow):**")
        for key, value in metrics.items():
            st.metric(key.replace("val_", ""), f"{value:.4f}")
    else:
        st.caption("Sem métricas de treino no MLflow para este modelo ainda (rode o retreinamento ou o backfill).")
    st.link_button("Abrir MLflow UI", MLFLOW_UI_URL, use_container_width=True)

tab_predict, tab_compare = st.tabs(["📄 Previsão em lote", "📊 Comparativo de modelos"])

with tab_predict:
    st.subheader("Envie um CSV ou parquet com as apólices a avaliar")
    st.caption(f"O arquivo precisa conter as {len(feature_columns)} colunas de feature usadas no treino (policy_id é opcional, mantido só para referência).")

    uploaded = st.file_uploader("Arquivo", type=["csv", "parquet"])

    if uploaded is not None:
        try:
            df_raw = _read_uploaded(uploaded)
        except Exception as exc:
            st.error(f"Não foi possível ler o arquivo: {exc}")
            df_raw = None

        if df_raw is not None:
            st.write(f"{len(df_raw)} linhas carregadas.")
            missing = [col for col in feature_columns if col not in df_raw.columns]
            if missing:
                st.error(f"Faltam {len(missing)} coluna(s) obrigatória(s): {missing}")
            else:
                with st.spinner("Calculando previsões..."):
                    try:
                        result = predict(df_raw, model_name, threshold=threshold)
                    except Exception as exc:
                        st.error(f"Falha ao gerar previsões: {exc}")
                        result = None

                if result is not None:
                    n_rows = len(result)
                    n_flagged = int(result["claim_prediction"].sum())

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Apólices avaliadas", n_rows)
                    col2.metric("Sinalizadas como sinistro", n_flagged)
                    col3.metric("Taxa sinalizada", f"{n_flagged / n_rows:.1%}" if n_rows else "—")

                    st.bar_chart(result["claim_probability"], height=200)
                    st.dataframe(result, use_container_width=True, height=400)

                    st.download_button(
                        "⬇️ Baixar previsões (CSV)",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name=f"predictions_{model_name.replace(' ', '_')}.csv",
                        mime="text/csv",
                    )

                    if log_to_mlflow:
                        run_id = log_batch_inference(model_name, n_rows, n_flagged, threshold)
                        if run_id:
                            st.success(f"Execução registrada no MLflow (run_id={run_id}).")
                        else:
                            st.warning("Não foi possível registrar no MLflow (servidor indisponível?).")

with tab_compare:
    st.subheader("Comparativo histórico dos modelos")
    ml_table = _load_results_table(PROJ_ROOT / "output" / "resultados_modelos.csv")
    glm_table = _load_results_table(PROJ_ROOT / "output" / "resultados_modelos_glm.csv")

    if ml_table is not None:
        st.markdown("**10 modelos de árvore/boosting** (`notebooks/model_ml.ipynb`)")
        st.dataframe(ml_table.round(4), use_container_width=True)
        st.bar_chart(ml_table.set_index("modelo")[["roc_auc", "pr_auc"]])
    else:
        st.caption("output/resultados_modelos.csv não encontrado ainda.")

    if glm_table is not None:
        st.markdown("**5 modelos GLM / generalistas** (`notebooks/glm_poisson.ipynb`)")
        st.dataframe(glm_table.round(4), use_container_width=True)
        st.bar_chart(glm_table.set_index("modelo")[["roc_auc", "pr_auc"]])

    st.caption(f"Para histórico completo de execuções e versões de modelo, veja o MLflow: {MLFLOW_UI_URL}")
