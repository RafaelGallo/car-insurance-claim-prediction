FROM python:3.12-slim

WORKDIR /srv/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY car_insurance_claim/ ./car_insurance_claim/
RUN pip install --no-cache-dir --no-deps -e .

COPY app/ ./app/

ENV MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MLFLOW_UI_URL=http://localhost:5000 \
    MLFLOW_EXPERIMENT_NAME=car_insurance_claim \
    PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
