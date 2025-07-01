"""Testes para avaliação do modelo"""
import json
from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent
METRICS_PATH = os.path.join(BASE_DIR, 'model', 'metrics.json')

def test_metrics_file_exists():
    assert os.path.exists(METRICS_PATH)

def test_metrics_content():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    assert 'mape' in metrics and 'rmse' in metrics and 'r2' in metrics
