"""
train.py — Fase 2: Treinamento e comparação de algoritmos de clustering.

Pipeline MLOps para segmentação de clientes bancários.
Compara KMeans, DBSCAN e Agglomerative Clustering com tracking completo via MLflow.

Autor: Leonardo Sampaio
Repositório: github.com/lsampaio/lsampaio-mlflow-segmentacao-clientes
"""

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Configuração ──────────────────────────────────────────────────────────────
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "segmentacao-clientes-bancarios"
FIGURES_DIR  = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "idade", "renda_mensal", "score_credito",
    "tempo_relacionamento_anos", "num_produtos",
    "ratio_renda_produtos", "score_normalizado", "valor_cliente",
]


# ── Utilitários ───────────────────────────────────────────────────────────────

def carregar_dados() -> np.ndarray:
    """Carrega o dataset processado pela Fase 1."""
    path = Path("data/clientes_features.csv")
    if not path.exists():
        print("Dataset não encontrado. Execute feature_engineering.py primeiro.")
        sys.exit(1)
    return pd.read_csv(path).values


def calcular_metricas(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calcula as 3 principais métricas de avaliação de clustering.

    Returns:
        dict com silhouette, davies_bouldin e calinski_harabasz
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {"silhouette": -1.0, "davies_bouldin": 99.0, "calinski_harabasz": 0.0}

    mask = labels != -1  # ignora outliers do DBSCAN
    return {
        "silhouette":          round(silhouette_score(X[mask], labels[mask]), 4),
        "davies_bouldin":      round(davies_bouldin_score(X[mask], labels[mask]), 4),
        "calinski_harabasz":   round(calinski_harabasz_score(X[mask], labels[mask]), 4),
    }


# ── Visualizações ─────────────────────────────────────────────────────────────

def plot_pca_clusters(X: np.ndarray, labels: np.ndarray, titulo: str) -> str:
    """Scatter plot 2D via PCA — mostra separação visual dos clusters."""
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X)
    variancia = pca.explained_variance_ratio_.sum() * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(X2d[:, 0], X2d[:, 1], c=labels,
                         cmap="tab10", alpha=0.75, s=40, edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"{titulo}\nPCA 2D — variância explicada: {variancia:.1f}%", fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.spines[["top", "right"]].set_visible(False)

    path = str(FIGURES_DIR / f"pca_{titulo.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_elbow_silhouette(X: np.ndarray) -> str:
    """Elbow curve + Silhouette para escolha do k ideal no KMeans."""
    k_range   = range(2, 11)
    inertias  = []
    silhuetas = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        silhuetas.append(silhouette_score(X, km.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(k_range, inertias, "bo-", linewidth=2, markersize=6)
    ax1.set_title("Elbow Curve — Inércia por k", fontsize=12)
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inércia")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(k_range, silhuetas, "go-", linewidth=2, markersize=6)
    ax2.set_title("Silhouette Score por k", fontsize=12)
    ax2.set_xlabel("Número de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.spines[["top", "right"]].set_visible(False)

    k_ideal = k_range[int(np.argmax(silhuetas))]
    ax2.axvline(k_ideal, color="red", linestyle="--", alpha=0.6,
                label=f"k ideal = {k_ideal}")
    ax2.legend()

    path = str(FIGURES_DIR / "kmeans_elbow_silhouette.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path, k_ideal


def plot_dendrograma(X: np.ndarray) -> str:
    """Dendrograma hierárquico — referência visual para Agglomerative."""
    fig, ax = plt.subplots(figsize=(14, 6))
    Z = linkage(X[:100], method="ward")   # amostra para legibilidade
    dendrogram(Z, ax=ax, truncate_mode="level", p=5,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Dendrograma — Agglomerative Clustering (Ward)\nAmostra de 100 clientes", fontsize=12)
    ax.set_xlabel("Índice do cliente")
    ax.set_ylabel("Distância")
    ax.spines[["top", "right"]].set_visible(False)

    path = str(FIGURES_DIR / "dendrograma_agglomerative.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ── Algoritmos ────────────────────────────────────────────────────────────────

def treinar_kmeans(X: np.ndarray):
    """
    KMeans com seleção automática do k ideal via Silhouette Score.
    Loga params, métricas, modelo e gráficos no MLflow.
    """
    mlflow.set_experiment(EXPERIMENT)

    # Determinar k ideal
    path_elbow, k_ideal = plot_elbow_silhouette(X)
    print(f"\n[KMeans] k ideal detectado: {k_ideal}")

    with mlflow.start_run(run_name=f"fase2_kmeans_k{k_ideal}"):
        modelo = KMeans(n_clusters=k_ideal, random_state=42, n_init=10)
        labels = modelo.fit_predict(X)
        metricas = calcular_metricas(X, labels)

        # Logging MLflow
        mlflow.log_param("algoritmo",    "KMeans")
        mlflow.log_param("n_clusters",   k_ideal)
        mlflow.log_param("n_init",       10)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics(metricas)
        mlflow.log_metric("inertia", round(modelo.inertia_, 2))
        mlflow.sklearn.log_model(modelo, name="modelo_kmeans")

        # Artefatos visuais
        mlflow.log_artifact(path_elbow)
        path_pca = plot_pca_clusters(X, labels, f"KMeans k={k_ideal}")
        mlflow.log_artifact(path_pca)

        print(f"  Silhouette:        {metricas['silhouette']}")
        print(f"  Davies-Bouldin:    {metricas['davies_bouldin']}")
        print(f"  Calinski-Harabasz: {metricas['calinski_harabasz']}")

    return labels, k_ideal


def treinar_dbscan(X: np.ndarray):
    """
    DBSCAN com grid search em eps e min_samples.
    Identifica automaticamente outliers e número de clusters.
    """
    mlflow.set_experiment(EXPERIMENT)

    grid = [(eps, ms) for eps in [0.3, 0.5, 0.8, 1.0]
                       for ms in [3, 5, 10]]
    melhor = {"silhouette": -1, "params": None, "labels": None}

    for eps, min_samples in grid:
        modelo = DBSCAN(eps=eps, min_samples=min_samples)
        labels = modelo.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue
        score = silhouette_score(X[labels != -1], labels[labels != -1])
        if score > melhor["silhouette"]:
            melhor = {"silhouette": score, "params": (eps, min_samples), "labels": labels}

    if melhor["params"] is None:
        print("[DBSCAN] Nenhuma configuração válida encontrada.")
        return None, None

    eps_best, ms_best = melhor["params"]
    labels = melhor["labels"]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()

    with mlflow.start_run(run_name=f"fase2_dbscan_eps{eps_best}_ms{ms_best}"):
        metricas = calcular_metricas(X, labels)

        mlflow.log_param("algoritmo",   "DBSCAN")
        mlflow.log_param("eps",         eps_best)
        mlflow.log_param("min_samples", ms_best)
        mlflow.log_metrics(metricas)
        mlflow.log_metric("n_clusters_encontrados", n_clusters)
        mlflow.log_metric("n_outliers",             int(n_outliers))
        mlflow.log_metric("pct_outliers",           round(n_outliers / len(labels) * 100, 2))

        path_pca = plot_pca_clusters(X, labels, f"DBSCAN eps={eps_best} ms={ms_best}")
        mlflow.log_artifact(path_pca)

        print(f"\n[DBSCAN] Melhor config: eps={eps_best}, min_samples={ms_best}")
        print(f"  Clusters encontrados: {n_clusters}")
        print(f"  Outliers: {n_outliers} ({n_outliers/len(labels)*100:.1f}%)")
        print(f"  Silhouette: {metricas['silhouette']}")

    return labels, (eps_best, ms_best)


def treinar_agglomerative(X: np.ndarray, n_clusters: int = 5):
    """
    Agglomerative Clustering com linkage Ward.
    Compara múltiplos métodos de linkage e loga dendrograma.
    """
    mlflow.set_experiment(EXPERIMENT)

    linkages = ["ward", "complete", "average"]
    melhor = {"silhouette": -1, "linkage": None, "labels": None}

    for lnk in linkages:
        modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=lnk)
        labels = modelo.fit_predict(X)
        score  = silhouette_score(X, labels)
        if score > melhor["silhouette"]:
            melhor = {"silhouette": score, "linkage": lnk, "labels": labels}

    labels   = melhor["labels"]
    lnk_best = melhor["linkage"]

    with mlflow.start_run(run_name=f"fase2_agglomerative_{lnk_best}_k{n_clusters}"):
        metricas = calcular_metricas(X, labels)

        mlflow.log_param("algoritmo",  "AgglomerativeClustering")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("linkage",    lnk_best)
        mlflow.log_metrics(metricas)

        path_dendro = plot_dendrograma(X)
        path_pca    = plot_pca_clusters(X, labels,
                                        f"Agglomerative {lnk_best} k={n_clusters}")
        mlflow.log_artifact(path_dendro)
        mlflow.log_artifact(path_pca)

        print(f"\n[Agglomerative] Melhor linkage: {lnk_best}, k={n_clusters}")
        print(f"  Silhouette:        {metricas['silhouette']}")
        print(f"  Davies-Bouldin:    {metricas['davies_bouldin']}")
        print(f"  Calinski-Harabasz: {metricas['calinski_harabasz']}")

    return labels


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fase 2 — Treinamento de algoritmos de clustering"
    )
    parser.add_argument(
        "--algoritmo",
        choices=["kmeans", "dbscan", "agglomerative", "todos"],
        default="todos",
        help="Algoritmo a executar (default: todos)",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Número de clusters para Agglomerative (default: 5)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)

    print("=" * 60)
    print("  Fase 2 — Segmentação de Clientes Bancários")
    print(f"  Algoritmo: {args.algoritmo}")
    print("=" * 60)

    X = carregar_dados()
    print(f"\nDataset carregado: {X.shape[0]} clientes, {X.shape[1]} features\n")

    if args.algoritmo in ("kmeans", "todos"):
        treinar_kmeans(X)

    if args.algoritmo in ("dbscan", "todos"):
        treinar_dbscan(X)

    if args.algoritmo in ("agglomerative", "todos"):
        treinar_agglomerative(X, n_clusters=args.n_clusters)

    print("\n" + "=" * 60)
    print("  Fase 2 concluída!")
    print(f"  Acesse a UI: http://127.0.0.1:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
