"""
Fase 1 — Feature engineering e análise exploratória.
Prepara o dataset para clustering e loga artefatos visuais no MLflow.
"""
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT  = "segmentacao-clientes-bancarios"


def tratar_outliers(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    df_clean = df.copy()
    for col in colunas:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = df_clean[col].clip(lower, upper)
    return df_clean


def criar_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ratio_renda_produtos"]   = df["renda_mensal"] / (df["num_produtos"] + 1)
    df["score_normalizado"]      = df["score_credito"] / 1000
    df["faixa_etaria"] = pd.cut(
        df["idade"],
        bins=[0, 25, 35, 45, 60, 100],
        labels=[1, 2, 3, 4, 5]
    ).astype(float)
    df["valor_cliente"] = (
        df["score_normalizado"] * 0.4 +
        (df["renda_mensal"] / df["renda_mensal"].max()) * 0.3 +
        (df["tempo_relacionamento_anos"] / df["tempo_relacionamento_anos"].max()) * 0.3
    )
    return df

def gerar_graficos_eda(df: pd.DataFrame, features: list) -> list:
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    artefatos = []

    # 1 — Heatmap de correlação
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Matriz de Correlação — Features de Clustering", fontsize=13)
    path = "reports/figures/heatmap_correlacao.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    artefatos.append(path)

    # 2 — Distribuição das features
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(features[:6]):
        axes[i].hist(df[col], bins=30, color="#378ADD", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].spines[["top", "right"]].set_visible(False)
    plt.suptitle("Distribuição das Features", fontsize=14, y=1.01)
    plt.tight_layout()
    path = "reports/figures/distribuicao_features.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    artefatos.append(path)

    # 3 — Boxplots por inadimplência
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, col in enumerate(["renda_mensal", "score_credito", "valor_cliente"]):
        df.boxplot(column=col, by="inadimplente", ax=axes[i],
                   boxprops=dict(color="#185FA5"),
                   medianprops=dict(color="#D85A30", linewidth=2))
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("Inadimplente (0=Não, 1=Sim)")
    plt.suptitle("Features por Status de Inadimplência", fontsize=13)
    plt.tight_layout()
    path = "reports/figures/boxplot_inadimplencia.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    artefatos.append(path)

    return artefatos


def pipeline_feature_engineering():
    from gerar_dados import gerar_dataset

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="fase1_feature_engineering"):
        # Gerar e salvar dados brutos
        df_raw = gerar_dataset()
        df_raw.to_csv("data/clientes_raw.csv", index=False)

        # Feature engineering
        features_numericas = [
            "idade", "renda_mensal", "score_credito",
            "tempo_relacionamento_anos", "num_produtos"
        ]
        df = tratar_outliers(df_raw, features_numericas)
        df = criar_features_derivadas(df)

        features_clustering = [
            "idade", "renda_mensal", "score_credito",
            "tempo_relacionamento_anos", "num_produtos",
            "ratio_renda_produtos", "score_normalizado", "valor_cliente"
        ]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features_clustering])
        df_scaled = pd.DataFrame(X_scaled, columns=features_clustering)
        df_scaled.to_csv("data/clientes_features.csv", index=False)

        # Logar parâmetros
        mlflow.log_param("n_amostras", len(df))
        mlflow.log_param("n_features", len(features_clustering))
        mlflow.log_param("outlier_method", "IQR_1.5x")
        mlflow.log_param("scaler", "StandardScaler")

        # Métricas básicas
        mlflow.log_metric("pct_inadimplentes", round(df["inadimplente"].mean() * 100, 2))
        mlflow.log_metric("renda_media", round(df["renda_mensal"].mean(), 2))
        mlflow.log_metric("score_medio", round(df["score_credito"].mean(), 2))

        # Gerar e logar gráficos EDA
        artefatos = gerar_graficos_eda(df, features_clustering)
        for artefato in artefatos:
            mlflow.log_artifact(artefato)

        print(f"Fase 1 concluída!")
        print(f"  Amostras: {len(df)} | Features: {len(features_clustering)}")
        print(f"  Inadimplentes: {df['inadimplente'].mean()*100:.1f}%")
        print(f"  Artefatos logados: {len(artefatos)}")

    return df_scaled, features_clustering


if __name__ == "__main__":
    pipeline_feature_engineering()
