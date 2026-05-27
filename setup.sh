#!/bin/bash
# =============================================================================
# setup.sh — Bootstrap do projeto na VM GCP
# Clona o repositório, instala dependências e sobe o MLflow server
#
# Uso:
#   chmod +x setup.sh && ./setup.sh
# =============================================================================

set -e  # para no primeiro erro

REPO_URL="https://github.com/leonardod38/lsampaio-mlflow-segmentacao-clientes.git"
BRANCH="feature/fase2-algoritmos-clustering"
PROJECT_DIR="$HOME/lsampaio-mlflow-segmentacao-clientes"
VENV_DIR="$HOME/mlflow-env"

echo "=============================================="
echo "  Setup — Segmentação de Clientes Bancários"
echo "=============================================="

# 1 — Clonar ou atualizar repositório
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "[1/5] Repositório já existe — atualizando..."
    cd "$PROJECT_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "[1/5] Clonando repositório..."
    git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# 2 — Ativar ambiente virtual existente
echo "[2/5] Ativando ambiente virtual..."
source "$VENV_DIR/bin/activate"

# 3 — Instalar/atualizar dependências
echo "[3/5] Verificando dependências..."
pip install -q mlflow scikit-learn pandas numpy matplotlib seaborn scipy

# 4 — Criar pastas necessárias
echo "[4/5] Criando estrutura de pastas..."
cd "$PROJECT_DIR"
mkdir -p data reports/figures

# 5 — Rodar pipeline completo (sempre da raiz do projeto)
echo "[5/5] Executando pipeline..."
echo ""
echo "▶ Fase 1 — Feature Engineering..."
python src/feature_engineering.py

echo ""
echo "▶ Fase 2 — Treinamento dos algoritmos..."
python src/train.py --algoritmo todos

echo ""
echo "▶ Fase 3 — Avaliação comparativa..."
python src/avaliar.py

echo ""
echo "=============================================="
echo "  Pipeline concluído!"
echo "  MLflow UI: http://$(curl -s ifconfig.me):5000"
echo "=============================================="
