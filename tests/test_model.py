"""
Testes para o modelo de previsão de ações
"""
import pytest
import joblib
import pandas as pd
from pathlib import Path
import os

# Configurações
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'prophet_model.joblib')

@pytest.fixture
def loaded_model():
    """Carrega o modelo para os testes"""
    return joblib.load(MODEL_PATH)

def test_model_loading(loaded_model):
    """Testa se o modelo carrega corretamente"""
    assert loaded_model is not None

def test_model_prediction(loaded_model):
    """Testa se o modelo faz previsões"""
    future = loaded_model.make_future_dataframe(periods=7)
    future['bias_adjust'] = 1.0
    future['Volume'] = 0.0
    future['event_peak'] = 0
    forecast = loaded_model.predict(future)
    assert len(forecast) > 0
    assert 'yhat' in forecast.columns
