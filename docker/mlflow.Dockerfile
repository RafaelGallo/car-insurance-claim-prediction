FROM python:3.12-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow==3.17.2

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "sqlite:////mlflow/data/mlflow.db", \
    "--default-artifact-root", "/mlflow/artifacts"]
