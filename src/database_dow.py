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
    raise FileNotFoundError(
        f"No credentials in / sem credenciais em: {config_dir}"
    )

# import only after the environment is set / importa só depois do ambiente definido
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

# authenticate on Kaggle / autentica no Kaggle
api = KaggleApi()
api.authenticate()