# Segmentação Inteligente de Clientes Bancários com MLflow

Pipeline MLOps completo para segmentação de clientes usando múltiplos algoritmos de clustering, com rastreamento de experimentos via MLflow rodando em VM no Google Cloud Platform.

## Sobre o projeto

Este projeto implementa um pipeline end-to-end de Machine Learning para segmentação de clientes bancários, cobrindo desde a geração e preparação dos dados até o serving do modelo via API REST.

**Contexto de negócio:** Identificar perfis de clientes para campanhas de marketing personalizadas, gestão de risco e oferta de produtos financeiros adequados.

## Stack tecnológica

- **MLflow 3.12** — tracking, registry e serving
- **scikit-learn** — KMeans, DBSCAN, AgglomerativeClustering
- **Python 3.14** — pipeline completo
- **Google Cloud Platform** — VM Compute Engine (e2-medium)
- **Ubuntu 26.04 LTS** — ambiente de produção
- **SQLite** — backend store do MLflow

## Estrutura do projeto

```
lsampaio-mlflow-segmentacao-clientes/
├── data/                    # Dados gerados e processados
├── src/                     # Scripts principais
│   ├── gerar_dados.py       # Geração do dataset sintético
│   ├── feature_engineering.py
│   ├── train.py             # Treinamento e tracking MLflow
│   └── avaliar.py           # Avaliação comparativa
├── notebooks/               # Análise exploratória
├── reports/
│   └── figures/             # Gráficos gerados
├── MLproject                # Definição do projeto MLflow
└── README.md
```

## Fases do projeto

### Fase 1 — Intermediário: Dados + Feature Engineering
- Dataset sintético com 300 clientes (idade, renda, score, produtos, inadimplência)
- Feature engineering: normalização, tratamento de outliers, features derivadas
- Análise exploratória com artefatos visuais no MLflow

### Fase 2 — Intermediário+: Múltiplos Algoritmos
- K-Means com otimização automática de k (Elbow + Silhouette)
- DBSCAN com grid search em eps e min_samples
- Agglomerative Clustering com dendrograma

### Fase 3 — Avançado: Avaliação e Visualização
- Matriz de métricas: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Visualizações: PCA 2D, t-SNE, radar chart por cluster
- Interpretação de negócio: nomeação de segmentos

### Fase 4 — Expert: MLOps Completo
- MLflow Projects com entrypoints parametrizados
- Model Registry com versionamento (Staging → Production)
- Model Serving via API REST

## Como executar

### Pré-requisitos
```bash
pip install mlflow scikit-learn pandas numpy matplotlib seaborn scipy
```

### Subir o MLflow server
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

### Executar pipeline completo
```bash
mlflow run . -P algoritmo=kmeans
mlflow run . -P algoritmo=dbscan
mlflow run . -P algoritmo=agglomerative
```

### Acessar a UI
```
http://localhost:5000
```

## Resultados

| Algoritmo | Melhor config | Silhouette | Davies-Bouldin |
|---|---|---|---|
| K-Means | k=5 | — | — |
| DBSCAN | eps=0.5 | — | — |
| Agglomerative | k=5, ward | — | — |

> Tabela será atualizada após execução completa do pipeline.

## Infraestrutura GCP

- **Máquina:** e2-medium (2 vCPU, 4GB RAM)
- **Região:** us-central1-a
- **SO:** Ubuntu 26.04 LTS Minimal
- **Custo estimado:** ~US$ 0,03/hora (desligada quando não utilizada)

## Autor

**Leonardo Sampaio**
ML Engineer | 20+ anos em tecnologia
Santander · Carrefour · Accenture

---
*Projeto desenvolvido como portfólio de MLOps — parte do plano de desenvolvimento para posições de ML Architect/Staff ML Engineer.*
