# CLAUDE.md — Memória do projeto

## Identidade do projeto
- **Nome:** lsampaio-mlflow-segmentacao-clientes
- **Autor:** Leonardo Sampaio (leonardod38)
- **Objetivo:** Pipeline MLOps completo de segmentação de clientes bancários
- **Repositório local:** E:\Claude\lsampaio-mlflow-segmentacao-clientes
- **VM GCP:** 35.184.43.161 (us-central1-a, e2-medium, Ubuntu 26.04 LTS)

## Stack
- Python 3.14
- MLflow 3.12
- scikit-learn, pandas, numpy, matplotlib, seaborn, scipy
- SQLite (backend MLflow)
- GCP Compute Engine

## Conexão GCP
```bash
# MobaXterm local terminal
gcp
# expande para:
ssh -F /dev/null -i ~/.ssh/mlflow-gcp leonardod38@35.184.43.161
```

## Subir MLflow server na VM
```bash
source ~/mlflow-env/bin/activate && \
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ~/mlflow-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*" > ~/mlflow.log 2>&1 &
```

## MLflow UI
```
http://35.184.43.161:5000
```

## Estrutura do projeto
```
lsampaio-mlflow-segmentacao-clientes/
├── data/
│   ├── clientes_raw.csv         # gerado por gerar_dados.py
│   └── clientes_features.csv    # gerado por feature_engineering.py
├── src/
│   ├── gerar_dados.py           # Fase 1 — dataset sintético (500 clientes)
│   ├── feature_engineering.py   # Fase 1 — EDA + features derivadas
│   ├── train.py                 # Fase 2 — KMeans, DBSCAN, Agglomerative
│   └── avaliar.py               # Fase 3 — métricas comparativas
├── notebooks/                   # análise exploratória
├── reports/
│   └── figures/                 # gráficos gerados
├── MLproject                    # entrypoints parametrizados
├── CLAUDE.md                    # este arquivo
├── .gitignore
└── README.md
```

## Fases e status
| Fase | Descrição | Status |
|------|-----------|--------|
| 1 | Dados + Feature Engineering | ✅ Concluída |
| 2 | Múltiplos algoritmos (KMeans, DBSCAN, Agglomerative) | 🔄 Em andamento |
| 3 | Avaliação profunda + visualizações avançadas | ⏳ Pendente |
| 4 | MLflow Projects + Registry + Serving | ⏳ Pendente |

## Experimento MLflow
- **Nome:** segmentacao-clientes-bancarios
- **Runs da Fase 1:** fase1_feature_engineering
- **Runs da Fase 2:** fase2_kmeans_k*, fase2_dbscan_*, fase2_agglomerative_*

## Features de clustering
```python
features_clustering = [
    "idade", "renda_mensal", "score_credito",
    "tempo_relacionamento_anos", "num_produtos",
    "ratio_renda_produtos", "score_normalizado", "valor_cliente"
]
```

## Segmentos de negócio esperados
| Cluster | Nome | Perfil |
|---------|------|--------|
| 0 | Cliente Premium | Alta renda, alto score, múltiplos produtos |
| 1 | Cliente Potencial | Renda média, score ok, pouco tempo |
| 2 | Cliente Básico | Baixa renda, score médio, 1-2 produtos |
| 3 | Cliente em Risco | Score baixo, inadimplência alta |
| 4 | Cliente Fidelizado | Tempo longo, score médio, poucos produtos |

## Decisões técnicas
- Outliers tratados com IQR 1.5x antes do scaling
- StandardScaler aplicado nas 8 features de clustering
- Silhouette score como métrica principal de comparação
- DBSCAN com grid search em eps=[0.3, 0.5, 0.8] e min_samples=[3, 5, 10]
- Agglomerative com linkage ward (melhor para clusters compactos)

## Git
```bash
cd E:\Claude\lsampaio-mlflow-segmentacao-clientes
git init
git add .
git commit -m "feat: fase 1 — dataset e feature engineering"
```

## Próximos passos
1. Criar src/train.py — Fase 2 (KMeans + DBSCAN + Agglomerative)
2. Criar src/avaliar.py — Fase 3 (métricas + visualizações avançadas)
3. Completar MLproject com entrypoints
4. Inicializar repositório GitHub público
5. Atualizar README com resultados reais
