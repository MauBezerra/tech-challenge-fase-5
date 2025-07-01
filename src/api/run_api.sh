#!/bin/bash

# Configuração padrão
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

uvicorn src.api.main:app \
    --host $HOST \
    --port $PORT \
    --reload
