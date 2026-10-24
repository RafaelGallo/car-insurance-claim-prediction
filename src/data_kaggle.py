"""Download Kaggle dataset and save as CSV, Parquet and SQLite.

Baixa o dataset do Kaggle e salva em CSV, Parquet e SQLite.
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# project folders / pastas do projeto
base_dir = Path(r"C:\Users\rafae\Documents\Projetos_data_science_IA\machine_learning\Car Insurance Claim Prediction")
input_dir = base_dir / "input"
data_dir = base_dir / "data"
sql_dir = base_dir / "sql"
config_dir = base_dir / "config"

# create output folders if missing / cria as pastas de saída se não existirem
for folder in tqdm((input_dir, data_dir, sql_dir), desc="Folders / pastas"):
    folder.mkdir(parents=True, exist_ok=True)

# point Kaggle to the local config folder before importing / aponta o Kaggle para a pasta local antes do import
os.environ["KAGGLE_CONFIG_DIR"] = str(config_dir)

# new API token file, if it exists / arquivo do novo token de API, se existir
token_file = config_dir / "access_token"
if token_file.exists():
    os.environ["KAGGLE_API_TOKEN"] = token_file.read_text(encoding="utf-8").strip()

# check that some credential exists / confere se existe alguma credencial
kaggle_json = config_dir / "kaggle.json"
if not kaggle_json.exists() and not token_file.exists():
    raise FileNotFoundError(f"No credentials in / sem credenciais em: {config_dir}")

# import only after the environment is set / importa só depois do ambiente definido
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

# authenticate on Kaggle / autentica no Kaggle
api = KaggleApi()
api.authenticate()

# dataset slug taken from the Kaggle URL / slug do dataset retirado da URL do Kaggle
dataset_slug = "ifteshanajnin/carinsuranceclaimprediction-classification"

# download and unzip with the built-in progress bar / baixa e descompacta com a barra de progresso nativa
api.dataset_download_files(dataset_slug, path=str(input_dir), unzip=True, quiet=False)

# list downloaded CSV files / lista os CSVs baixados
csv_files = sorted(input_dir.glob("*.csv"))
print("CSV files / arquivos CSV:", [f.name for f in csv_files])

# rows written per SQLite chunk / linhas gravadas por bloco no SQLite
chunk_size = 10_000

# open SQLite connection / abre a conexão SQLite
sqlite_path = sql_dir / "car_insurance_claim.db"
conn = sqlite3.connect(sqlite_path)

# convert each CSV to Parquet and SQLite table / converte cada CSV para Parquet e tabela SQLite
for csv_path in tqdm(csv_files, desc="Files / arquivos"):
    table_name = csv_path.stem

    # read CSV / lê o CSV
    df = pd.read_csv(csv_path)

    # save Parquet in the data folder / salva o Parquet na pasta data
    parquet_path = data_dir / f"{table_name}.parquet"
    df.to_parquet(parquet_path, index=False)

    # write to SQLite in chunks, first chunk replaces the table / grava no SQLite em blocos, o primeiro substitui a tabela
    for start in tqdm(
        range(0, len(df), chunk_size),
        desc=f"SQLite {table_name}",
        leave=False,
    ):
        df.iloc[start:start + chunk_size].to_sql(
            table_name,
            conn,
            if_exists="replace" if start == 0 else "append",
            index=False,
        )

    # quick summary / resumo rápido
    tqdm.write(f"{table_name}: {df.shape[0]} rows / linhas, {df.shape[1]} columns / colunas")

# validate saved tables in SQLite / valida as tabelas salvas no SQLite
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type = 'table'", conn)
print("SQLite tables / tabelas SQLite:", tables["name"].tolist())

# validate row counts per table / valida a contagem de linhas por tabela
for table_name in tqdm(tables["name"], desc="Validation / validação"):
    n_rows = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table_name}", conn)["n"].iloc[0]
    tqdm.write(f"{table_name}: {n_rows} rows / linhas")

# close SQLite connection / fecha a conexão SQLite
conn.close()

# validate saved files / valida os arquivos salvos
print("Parquet files / arquivos Parquet:", sorted(p.name for p in data_dir.glob("*.parquet")))
print("SQLite file / arquivo SQLite:", sqlite_path.name, f"({sqlite_path.stat().st_size / 1e6:.1f} MB)")