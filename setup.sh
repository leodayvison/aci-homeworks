#!/usr/bin/env bash
set -e

# Cria o ambiente virtual caso ainda não exista
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Instala as dependências usando o pip isolado do venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo 'Ambiente configurado com sucesso! Lembre de sempre iniciar o ambiente virtual digitando "source .venv/bin/activate" e desativar ao terminar o trabalho, digitando "deactivate"'
