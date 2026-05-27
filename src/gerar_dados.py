"""
Fase 1 — Geração do dataset sintético de clientes bancários.
Simula distribuições realistas baseadas em padrões de mercado financeiro.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 500

def gerar_dataset():
    # Segmentos base para distribuições realistas
    segmento = np.random.choice(
        ["premium", "potencial", "basico", "risco"],
        size=N,
        p=[0.15, 0.30, 0.40, 0.15]
    )

    idade = np.where(segmento == "premium",
        np.random.normal(48, 8, N),
        np.where(segmento == "potencial",
            np.random.normal(35, 7, N),
            np.where(segmento == "basico",
                np.random.normal(30, 9, N),
                np.random.normal(38, 10, N)
            )
        )
    ).clip(18, 80)

    renda_mensal = np.where(segmento == "premium",
        np.random.normal(18000, 5000, N),
        np.where(segmento == "potencial",
            np.random.normal(7000, 2000, N),
            np.where(segmento == "basico",
                np.random.normal(2500, 800, N),
                np.random.normal(3000, 1200, N)
            )
        )
    ).clip(1200, 60000)

    score_credito = np.where(segmento == "premium",
        np.random.normal(820, 40, N),
        np.where(segmento == "potencial",
            np.random.normal(700, 60, N),
            np.where(segmento == "basico",
                np.random.normal(580, 80, N),
                np.random.normal(450, 90, N)
            )
        )
    ).clip(300, 1000)

    tempo_relacionamento = np.where(segmento == "premium",
        np.random.normal(12, 4, N),
        np.where(segmento == "potencial",
            np.random.normal(5, 3, N),
            np.where(segmento == "basico",
                np.random.normal(3, 2, N),
                np.random.normal(4, 3, N)
            )
        )
    ).clip(0, 30)

    num_produtos = np.where(segmento == "premium",
        np.random.randint(4, 8, N),
        np.where(segmento == "potencial",
            np.random.randint(2, 5, N),
            np.where(segmento == "basico",
                np.random.randint(1, 3, N),
                np.random.randint(1, 4, N)
            )
        )
    )

    inadimplente = np.where(segmento == "risco",
        np.random.binomial(1, 0.55, N),
        np.where(segmento == "basico",
            np.random.binomial(1, 0.15, N),
            np.random.binomial(1, 0.03, N)
        )
    )

    df = pd.DataFrame({
        "idade": idade.astype(int),
        "renda_mensal": renda_mensal.round(2),
        "score_credito": score_credito.astype(int),
        "tempo_relacionamento_anos": tempo_relacionamento.round(1),
        "num_produtos": num_produtos,
        "inadimplente": inadimplente,
        "segmento_real": segmento
    })

    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = gerar_dataset()
    df.to_csv("data/clientes_raw.csv", index=False)
    print(f"Dataset gerado: {df.shape[0]} clientes, {df.shape[1]} features")
    print(f"\nDistribuição por segmento:\n{df['segmento_real'].value_counts()}")
    print(f"\nEstatísticas descritivas:\n{df.describe().round(2)}")
