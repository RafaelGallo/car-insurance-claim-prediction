FROM apache/airflow:2.10.5-python3.12

USER airflow

COPY airflow/requirements-dag.txt /requirements-dag.txt
RUN pip install --no-cache-dir -r /requirements-dag.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"

COPY --chown=airflow:root pyproject.toml README.md LICENSE /opt/car_insurance_claim/
COPY --chown=airflow:root car_insurance_claim/ /opt/car_insurance_claim/car_insurance_claim/
RUN pip install --no-cache-dir --no-deps -e /opt/car_insurance_claim
