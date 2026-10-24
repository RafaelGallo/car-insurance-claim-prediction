from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from car_insurance_claim.config import DOC_DIR, IMG_DIR, PROJ_ROOT

app = typer.Typer()

PAGE_WIDTH, _ = A4
CONTENT_WIDTH = PAGE_WIDTH - 4 * cm


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Caption", parent=styles["Italic"], fontSize=9, textColor=colors.grey, spaceAfter=14))
    styles.add(ParagraphStyle("CodeBlock", parent=styles["Normal"], fontName="Courier", fontSize=8.5, leading=11, backColor=colors.whitesmoke))
    return styles


STYLES = _styles()


def _fig(path: Path, caption: str) -> list:
    with PILImage.open(path) as im:
        w, h = im.size
    width = CONTENT_WIDTH
    height = width * h / w
    max_height = 16 * cm
    if height > max_height:
        height = max_height
        width = height * w / h
    return [Image(str(path), width=width, height=height), Paragraph(caption, STYLES["Caption"])]


def _metrics_table(df: pd.DataFrame, columns: list, first_col_width=None) -> Table:
    header = [c for c in columns if c in df.columns]
    celula = ParagraphStyle("Celula", parent=STYLES["Normal"], fontSize=7.5, leading=9)
    data = [header]
    for _, row in df.iterrows():
        linha = []
        for c in header:
            valor = row[c]
            if isinstance(valor, float):
                linha.append(f"{valor:.4f}")
            elif c == header[0]:
                linha.append(Paragraph(str(valor), celula))
            else:
                linha.append(str(valor))
        data.append(linha)
    widths = [first_col_width] + [None] * (len(header) - 1) if first_col_width else None
    table = Table(data, hAlign="LEFT", colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]
        )
    )
    return table


@app.command()
def main(output_path: Path = typer.Option(DOC_DIR / "car_insurance_claim_report.pdf")):
    styles = STYLES
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Car Insurance Claim Risk Prediction",
    )

    story = []

    # --- capa ---
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("Car Insurance Claim Risk Prediction", styles["Title"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Relatório técnico — EDA, modelagem (GLM e árvore/boosting) e arquitetura MLOps", styles["Heading2"]))
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("Gerado automaticamente a partir de notebooks/, output/ e img/.", styles["Normal"]))
    story.append(PageBreak())

    # --- problema de negocio ---
    story.append(Paragraph("1. Problema de negócio", styles["Heading1"]))
    story.append(
        Paragraph(
            "Uma seguradora precisa decidir, na emissão e renovação da apólice, qual a probabilidade de cada "
            "segurado acionar o seguro nos próximos 6 meses. O score apoia precificação baseada em risco, "
            "priorização de apólices para revisão manual/triagem de fraude e redução de sinistralidade sem "
            "rejeitar bons clientes. A base (Kaggle Car Insurance Claim Prediction) tem 58.592 apólices e 44 "
            "colunas — perfil do segurado, características do veículo e o alvo binário <b>is_claim</b> — com "
            "forte desbalanceamento: apenas 6,4% das apólices acionam o seguro em 6 meses.",
            styles["Normal"],
        )
    )
    story.append(PageBreak())

    # --- insights EDA ---
    story.append(Paragraph("2. Principais insights (EDA)", styles["Heading1"]))
    insights = [
        ("output1.png", "Alvo raro e desbalanceado: só 6,4% das apólices (3.748 de 58.592) acionam o seguro em 6 meses."),
        ("2output.png", "policy_tenure é a variável com maior correlação com is_claim (+0,08), ainda fraca. As variáveis de porte/motor do veículo são fortemente correlacionadas entre si (até 0,96), mas quase 0 com o alvo."),
        ("4output.png", "Tempo de apólice é o sinal mais forte: apólices longas/muito longas têm taxa de sinistro de 8,3-8,7%, mais que o dobro das muito curtas (3,6%)."),
        ("5output.png", "Risco varia por região: os clusters C14, C4 e C19 têm taxa de sinistro acima de 7,4% vs. 6,4% da carteira."),
        ("6output.png", "Itens de segurança eletrônicos (sensor de estacionamento, câmera de ré, ESC, TPMS) não reduzem a taxa bruta de sinistro."),
        ("7output.png", "Outliers tratados por z-score (limitados a 3 desvios-padrão, sem remover apólices) em age_of_car, age_of_policyholder e population_density."),
    ]
    for filename, caption in insights:
        path = IMG_DIR / filename
        if path.exists():
            story.extend(_fig(path, caption))
            story.append(Spacer(1, 0.4 * cm))
    story.append(PageBreak())

    # --- GLM ---
    story.append(Paragraph("3. Abordagem estatística (GLM / Poisson)", styles["Heading1"]))
    for texto in [
        "notebooks/glm_poisson.ipynb compara 5 modelos generalistas da família GLM (statsmodels), em um for: "
        "Poisson, Binomial-Logit, Binomial-Probit, Binomial-Cloglog e Binomial Negativa.",
        "<b>Protocolo sem vazamento.</b> A divisão treino/teste (80/20) vem antes de tudo. Limites de outlier, "
        "correlações, categorias das dummies, escolha de variáveis e limiar de decisão vêm só do treino; as escolhas "
        "são feitas por validação cruzada 5-fold no treino e o teste só é reportado. Nenhuma coluna nova é criada "
        "na base: só seleção de colunas existentes e codificação 0/1 das categóricas e Sim/Não.",
        "<b>Variáveis.</b> policy_tenure (maior correlação positiva com o alvo), age_of_policyholder, age_of_car "
        "(correlação negativa, mas 2º sinal mais forte), cylinder e is_parking_sensors (veículo: correlação positiva "
        "acima de 2/raiz(n), sem colinearidade e com informação nova), segment, transmission_type e area_cluster.",
        "<b>Resultado.</b> A versão inicial não dava policy_tenure aos 4 modelos não-Poisson; corrigido isso, eles vão "
        "de ROC-AUC 0,546 para 0,610 na validação cruzada e de 0,555 para 0,586 no teste. O Poisson praticamente "
        "não muda. As 5 famílias ficam indistinguíveis. Marcando 45% da carteira, o modelo pega ~55% dos sinistros "
        "com precisão de ~8% (taxa base 6,4%).",
        "<b>Limite.</b> Uma árvore com as mesmas colunas do GLM chega a ~0,65: a diferença é a forma linear, não "
        "falta de informação. O caso mais claro é age_of_car (carros com idade 0 têm ~10% de sinistro, contra "
        "4,5-7% nas outras faixas); representar isso exige uma coluna nova.",
    ]:
        story.append(Paragraph(texto, styles["Normal"]))
        story.append(Spacer(1, 0.2 * cm))
    glm_csv = PROJ_ROOT / "output" / "resultados_modelos_glm.csv"
    if glm_csv.exists():
        glm_df = pd.read_csv(glm_csv)
        story.append(_metrics_table(glm_df, ["modelo", "roc_auc", "pr_auc", "recall", "precisao", "sinalizadas", "cv_roc_auc"]))
    comparativo_csv = PROJ_ROOT / "output" / "comparativo_glm.csv"
    if comparativo_csv.exists():
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("Antes vs. depois e referências (mesmo protocolo)", styles["Heading3"]))
        comp_df = pd.read_csv(comparativo_csv)
        story.append(
            _metrics_table(comp_df, ["configuracao", "cv_roc_auc", "teste_roc_auc", "teste_pr_auc", "recall_corte_20"], first_col_width=7.5 * cm)
        )
    story.append(Spacer(1, 0.6 * cm))
    for filename, caption in [
        ("7output_glm_correlacao_veiculo.png", "Correlação entre as variáveis numéricas do veículo (treino)."),
        ("8output_glm_correlacao_alvo.png", "Correlação de cada variável com is_claim no treino (X-Y); tracejado = ±2/raiz(n)."),
        ("9output_glm_roc.png", "Curva ROC dos 5 modelos GLM na base de teste."),
        ("10output_glm_confusion.png", "Matrizes de confusão dos 5 modelos GLM, todos marcando 45% da carteira (corte definido no treino)."),
    ]:
        path = IMG_DIR / filename
        if path.exists():
            story.extend(_fig(path, caption))
            story.append(Spacer(1, 0.4 * cm))
    story.append(PageBreak())

    # --- modelos de producao ---
    story.append(Paragraph("4. Modelos de produção (árvore/boosting)", styles["Heading1"]))
    story.append(
        Paragraph(
            "notebooks/model_ml.ipynb treina e compara 10 modelos (mesmas colunas, label-encoded) e "
            "notebooks/pipeline_ml.ipynb ajusta 2 pipelines HistGradientBoosting tunados (RandomizedSearchCV / "
            "GridSearchCV). Os artefatos de produção (.joblib) ficam em models/. O limiar de negócio padrão é "
            "<b>0,55</b> de probabilidade de sinistro (models/limiar_negocio.joblib).",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    ml_csv = PROJ_ROOT / "output" / "resultados_modelos.csv"
    if ml_csv.exists():
        ml_df = pd.read_csv(ml_csv).sort_values("roc_auc", ascending=False)
        story.append(_metrics_table(ml_df, ["modelo", "roc_auc", "pr_auc", "acuracia", "f1"]))
    story.append(PageBreak())

    # --- arquitetura MLOps ---
    story.append(Paragraph("5. Arquitetura MLOps", styles["Heading1"]))
    story.append(
        Paragraph(
            "car_insurance_claim/ é o pacote compartilhado por todos os componentes: modeling/predict.py "
            "(inferência — carrega models/, aplica os mesmos LabelEncoders do treino, mapeia categorias não "
            "vistas, aplica o limiar de negócio), modeling/train.py (retreino dos 10 modelos + log no MLflow, "
            "chamado pelo Airflow) e mlops/ (utilitários de MLflow e backfill de métricas históricas).",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    componentes = [
        ["Componente", "Porta", "Papel"],
        ["Streamlit (app/)", "8501", "Seleção de modelo, upload de CSV/parquet, previsão em lote, comparativo de modelos"],
        ["MLflow (docker/mlflow.Dockerfile)", "5000", "Tracking de métricas, registro de modelos, histórico de execuções"],
        ["Airflow (airflow/)", "8080", "DAG semanal: retreina os 10 modelos a partir de data/raw e loga no MLflow"],
        ["Postgres", "interno", "Metadata store do Airflow"],
    ]
    tbl = Table(componentes, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("6. Como executar", styles["Heading2"]))
    story.append(
        Paragraph(
            "docker compose up -d --build  (Streamlit + MLflow)<br/>"
            "docker compose exec streamlit python -m car_insurance_claim.mlops.backfill  (popula o histórico do MLflow)<br/>"
            "docker compose -f docker-compose.airflow.yml up -d --build  (Airflow, opcional)",
            styles["CodeBlock"],
        )
    )

    doc.build(story)
    logger.success(f"Relatório gerado em {output_path}")


if __name__ == "__main__":
    app()
