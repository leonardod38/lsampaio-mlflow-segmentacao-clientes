"""
avaliar.py — Fase 3: Avaliação comparativa e visualizações avançadas.

Compara todos os algoritmos via 3 métricas, gera radar chart por cluster
e tabela de interpretação de negócio.

Autor: Leonardo Sampaio
"""

import sys
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "segmentacao-clientes-bancarios"
FIGURES_DIR  = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "idade", "renda_mensal", "score_credito",
    "tempo_relacionamento_anos", "num_produtos",
    "ratio_renda_produtos", "score_normalizado", "valor_cliente",
]

NOMES_SEGMENTOS = {
    0: "Cliente Premium",
    1: "Cliente Potencial",
    2: "Cliente Básico",
    3: "Cliente em Risco",
    4: "Cliente Fidelizado",
}


def carregar_dados():
    path = Path("data/clientes_features.csv")
    if not path.exists():
        print("Execute feature_engineering.py primeiro.")
        sys.exit(1)
    return pd.read_csv(path).values


def calcular_metricas(X, labels):
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2:
        return {"silhouette": -1.0, "davies_bouldin": 99.0, "calinski_harabasz": 0.0}
    return {
        "silhouette":        round(silhouette_score(X[mask], labels[mask]), 4),
        "davies_bouldin":    round(davies_bouldin_score(X[mask], labels[mask]), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X[mask], labels[mask]), 4),
    }


def plot_radar_cluster(df_raw, labels, titulo):
    """Radar chart mostrando o perfil médio de cada cluster."""
    cols = ["idade", "renda_mensal", "score_credito",
            "tempo_relacionamento_anos", "num_produtos"]
    df_raw = pd.read_csv("data/clientes_raw.csv")
    df_raw["cluster"] = labels

    clusters = sorted(set(labels[labels != -1]))
    medias = df_raw[df_raw["cluster"] != -1].groupby("cluster")[cols].mean()
    medias_norm = (medias - medias.min()) / (medias.max() - medias.min() + 1e-9)

    N = len(cols)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    for i, cluster in enumerate(clusters):
        values = medias_norm.loc[cluster].tolist()
        values += values[:1]
        nome = NOMES_SEGMENTOS.get(cluster, f"Cluster {cluster}")
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i], label=nome)
        ax.fill(angles, values, alpha=0.12, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Idade", "Renda", "Score", "Tempo Rel.", "Produtos"],
                       fontsize=10)
    ax.set_title(f"Perfil dos Segmentos — {titulo}\n", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    path = str(FIGURES_DIR / f"radar_{titulo.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_comparacao_metricas(resultados):
    """Heatmap comparativo das 3 métricas entre algoritmos."""
    df = pd.DataFrame(resultados).T
    df_plot = df[["silhouette", "davies_bouldin", "calinski_harabasz"]].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metricas = {
        "silhouette":        ("Silhouette Score",       "Maior = melhor", "Greens"),
        "davies_bouldin":    ("Davies-Bouldin Score",   "Menor = melhor", "Reds_r"),
        "calinski_harabasz": ("Calinski-Harabasz Score","Maior = melhor", "Blues"),
    }

    for ax, (col, (titulo, subtitulo, cmap)) in zip(axes, metricas.items()):
        vals = df_plot[[col]]
        im = ax.imshow(vals.values, cmap=cmap, aspect="auto")
        ax.set_xticks([0])
        ax.set_xticklabels([col.replace("_", " ").title()], fontsize=9)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=9)
        for i, v in enumerate(vals.values.flatten()):
            ax.text(0, i, f"{v:.4f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
        ax.set_title(f"{titulo}\n{subtitulo}", fontsize=10)

    plt.suptitle("Comparação de Algoritmos — Métricas de Clustering", fontsize=13, y=1.02)
    plt.tight_layout()
    path = str(FIGURES_DIR / "comparacao_metricas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def gerar_tabela_negocios(df_raw_path, labels, algoritmo):
    """Gera CSV com perfil médio por segmento para interpretação de negócio."""
    df = pd.read_csv(df_raw_path)
    df["cluster"] = labels
    df["segmento"] = df["cluster"].map(
        lambda x: NOMES_SEGMENTOS.get(x, f"Cluster {x}") if x != -1 else "Outlier"
    )
    cols = ["idade", "renda_mensal", "score_credito",
            "tempo_relacionamento_anos", "num_produtos", "inadimplente"]
    resumo = df.groupby("segmento")[cols].mean().round(2)
    resumo["total_clientes"] = df.groupby("segmento").size()
    resumo["pct_base"] = (resumo["total_clientes"] / len(df) * 100).round(1)
    path = f"reports/tabela_segmentos_{algoritmo}.csv"
    resumo.to_csv(path)
    return path, resumo


def pipeline_avaliacao(X):
    """Roda os 3 algoritmos, compara métricas e loga tudo no MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    algoritmos = {
        "KMeans_k5":          KMeans(n_clusters=5, random_state=42, n_init=10),
        "DBSCAN":             DBSCAN(eps=0.5, min_samples=5),
        "Agglomerative_Ward": AgglomerativeClustering(n_clusters=5, linkage="ward"),
    }

    resultados = {}

    with mlflow.start_run(run_name="fase3_avaliacao_comparativa"):
        for nome, modelo in algoritmos.items():
            labels = modelo.fit_predict(X)
            metricas = calcular_metricas(X, labels)
            resultados[nome] = metricas

            # Log por algoritmo
            for k, v in metricas.items():
                mlflow.log_metric(f"{nome}_{k}", v)

            # Radar chart
            path_radar = plot_radar_cluster(None, labels, nome)
            mlflow.log_artifact(path_radar)

            # Tabela de negócio
            path_csv, resumo = gerar_tabela_negocios(
                "data/clientes_raw.csv", labels, nome
            )
            mlflow.log_artifact(path_csv)

            print(f"\n[{nome}]")
            print(f"  Silhouette:        {metricas['silhouette']}")
            print(f"  Davies-Bouldin:    {metricas['davies_bouldin']}")
            print(f"  Calinski-Harabasz: {metricas['calinski_harabasz']}")
            print(resumo[["total_clientes", "pct_base", "renda_mensal", "score_credito"]].to_string())

        # Heatmap comparativo
        path_comp = plot_comparacao_metricas(resultados)
        mlflow.log_artifact(path_comp)

        # Ranking final
        df_rank = pd.DataFrame(resultados).T
        melhor = df_rank["silhouette"].idxmax()
        mlflow.log_param("melhor_algoritmo", melhor)

        print(f"\n{'='*55}")
        print(f"  Melhor algoritmo: {melhor}")
        print(f"  Silhouette: {df_rank.loc[melhor, 'silhouette']}")
        print(f"{'='*55}")

    return resultados


if __name__ == "__main__":
    X = carregar_dados()
    print(f"Dataset: {X.shape[0]} clientes, {X.shape[1]} features\n")
    pipeline_avaliacao(X)
    print("\nFase 3 concluída! Acesse http://127.0.0.1:5000")
