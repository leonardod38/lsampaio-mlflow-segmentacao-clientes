"""
registry.py — Fase 4: Model Registry + Versionamento + Serving via API REST.

Registra o melhor modelo no MLflow Registry, promove para Production
e demonstra inferência via API REST.

Autor: Leonardo Sampaio
"""

import sys
import json
import subprocess
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from mlflow import MlflowClient
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ── Configuração ──────────────────────────────────────────────────────────────
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "segmentacao-clientes-bancarios"
MODEL_NAME   = "segmentacao-clientes-bancarios"
ROOT         = Path(__file__).resolve().parent.parent

FEATURES = [
    "idade", "renda_mensal", "score_credito",
    "tempo_relacionamento_anos", "num_produtos",
    "ratio_renda_produtos", "score_normalizado", "valor_cliente",
]

SEGMENTOS = {
    0: "Cliente Premium",
    1: "Cliente Potencial",
    2: "Cliente Básico",
    3: "Cliente em Risco",
    4: "Cliente Fidelizado",
   -1: "Outlier / Indefinido",
}


# ── Pipeline de registro ───────────────────────────────────────────────────────

def treinar_modelo_producao(X: np.ndarray) -> tuple:
    """Treina o DBSCAN com os melhores parâmetros e retorna modelo + labels."""
    modelo = DBSCAN(eps=0.5, min_samples=5)
    labels = modelo.fit_predict(X)
    return modelo, labels


def registrar_modelo(X: np.ndarray) -> str:
    """
    Treina, loga e registra o modelo no MLflow Model Registry.
    Retorna o run_id do modelo registrado.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    print("\n[1/4] Treinando modelo de produção (DBSCAN)...")
    modelo, labels = treinar_modelo_producao(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers  = int((labels == -1).sum())

    with mlflow.start_run(run_name="fase4_modelo_producao") as run:
        # Parâmetros
        mlflow.log_param("algoritmo",    "DBSCAN")
        mlflow.log_param("eps",          0.5)
        mlflow.log_param("min_samples",  5)
        mlflow.log_param("ambiente",     "producao")

        # Métricas
        mlflow.log_metric("n_clusters",  n_clusters)
        mlflow.log_metric("n_outliers",  n_outliers)
        mlflow.log_metric("pct_outliers", round(n_outliers / len(labels) * 100, 2))

        # Registrar modelo com nome
        mlflow.sklearn.log_model(
            modelo,
            name="modelo_dbscan_producao",
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")
        print(f"  Clusters: {n_clusters} | Outliers: {n_outliers}")

    return run_id


def promover_para_producao() -> None:
    """
    Busca a versão mais recente do modelo e promove para Production.
    Arquiva versões anteriores automaticamente.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)

    print("\n[2/4] Promovendo modelo para Production...")

    # Buscar versões do modelo
    versoes = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versoes:
        print("  Nenhum modelo encontrado no Registry.")
        return

    # Pegar a versão mais recente
    versao_mais_recente = max(versoes, key=lambda v: int(v.version))
    versao_num = versao_mais_recente.version

    # Adicionar tags descritivas
    client.set_model_version_tag(
        MODEL_NAME, versao_num, "algoritmo", "DBSCAN"
    )
    client.set_model_version_tag(
        MODEL_NAME, versao_num, "ambiente", "producao"
    )
    client.set_model_version_tag(
        MODEL_NAME, versao_num, "aprovado_por", "Leonardo Sampaio"
    )

    # Atualizar alias para production
    client.set_registered_model_alias(MODEL_NAME, "production", versao_num)

    print(f"  Versão {versao_num} promovida para Production ✅")
    print(f"  Alias 'production' → versão {versao_num}")


def carregar_modelo_producao():
    """Carrega o modelo em Production direto do Registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{MODEL_NAME}@production"
    print(f"\n[3/4] Carregando modelo: {uri}")
    modelo = mlflow.sklearn.load_model(uri)
    print("  Modelo carregado com sucesso ✅")
    return modelo


def inferir_clientes(modelo, scaler: StandardScaler) -> None:
    """
    Demonstra inferência com 5 perfis de clientes reais.
    Mostra o segmento predito para cada um.
    """
    print("\n[4/4] Inferência — classificando novos clientes...")
    print("=" * 60)

    clientes_novos = [
        {"nome": "Carlos",  "idade": 52, "renda_mensal": 22000, "score_credito": 850,
         "tempo_relacionamento_anos": 15, "num_produtos": 6},
        {"nome": "Ana",     "idade": 28, "renda_mensal": 4500,  "score_credito": 680,
         "tempo_relacionamento_anos": 2,  "num_produtos": 2},
        {"nome": "Pedro",   "idade": 35, "renda_mensal": 2200,  "score_credito": 420,
         "tempo_relacionamento_anos": 1,  "num_produtos": 1},
        {"nome": "Maria",   "idade": 45, "renda_mensal": 8000,  "score_credito": 720,
         "tempo_relacionamento_anos": 10, "num_produtos": 4},
        {"nome": "Roberto", "idade": 31, "renda_mensal": 1800,  "score_credito": 380,
         "tempo_relacionamento_anos": 3,  "num_produtos": 2},
    ]

    for c in clientes_novos:
        # Feature engineering básico
        ratio  = c["renda_mensal"] / (c["num_produtos"] + 1)
        score_n = c["score_credito"] / 1000
        valor   = score_n * 0.4 + (c["renda_mensal"] / 25000) * 0.3 + (c["tempo_relacionamento_anos"] / 20) * 0.3

        vetor = [[
            c["idade"], c["renda_mensal"], c["score_credito"],
            c["tempo_relacionamento_anos"], c["num_produtos"],
            ratio, score_n, valor
        ]]

        X_scaled = scaler.transform(vetor)
        cluster  = modelo.fit_predict(X_scaled)[0]
        segmento = SEGMENTOS.get(int(cluster), f"Cluster {cluster}")

        print(f"  {c['nome']:10} | Renda: R${c['renda_mensal']:>7,.0f} | "
              f"Score: {c['score_credito']} → {segmento}")

    print("=" * 60)


def pipeline_registry():
    """Executa o pipeline completo da Fase 4."""
    mlflow.set_tracking_uri(TRACKING_URI)

    # Carregar dados processados
    path = ROOT / "data" / "clientes_features.csv"
    if not path.exists():
        print("Execute feature_engineering.py primeiro.")
        sys.exit(1)

    X = pd.read_csv(path).values

    # Carregar scaler original (recriamos para inferência)
    df_raw  = pd.read_csv(ROOT / "data" / "clientes_raw.csv")
    scaler  = StandardScaler()
    cols    = ["idade", "renda_mensal", "score_credito",
               "tempo_relacionamento_anos", "num_produtos"]
    scaler.fit(df_raw[cols])

    # Executar pipeline
    registrar_modelo(X)
    promover_para_producao()
    modelo = carregar_modelo_producao()
    inferir_clientes(modelo, scaler)

    print(f"\nFase 4 concluída!")
    print(f"Model Registry: http://127.0.0.1:5000/#/models/{MODEL_NAME}")


if __name__ == "__main__":
    pipeline_registry()
