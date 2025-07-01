"""
API FastAPI para previsão de preços de ações
"""
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import os
from datetime import datetime

# Configurações
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'prophet_model.joblib')

app = FastAPI(title="Stock Prediction API",
              description="API para previsão de preços de ações usando Prophet")

# Instrumentação Prometheus
Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

class PredictionRequest(BaseModel):
    days: int = 7

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Endpoint para previsão de preços
    
    Args:
        days: Número de dias para prever (1-30)
    
    Returns:
        Lista de previsões com datas e valores
    """
    if not 1 <= request.days <= 30:
        raise HTTPException(status_code=400, detail="Número de dias deve ser entre 1 e 30")
        
    try:
        # Carregar modelo
        model = joblib.load(MODEL_PATH)
        
        # Gerar previsões
        future = model.make_future_dataframe(periods=request.days)
        future['bias_adjust'] = 1.0  # regressor obrigatório
        # Volume: usa último valor do histórico
        volume_default = model.history['Volume'].rolling(5).mean().iloc[-1] if 'Volume' in model.history else 0.0
        future['Volume'] = volume_default
        future['event_peak'] = 0
        forecast = model.predict(future)
        
        # Formatar resposta
        predictions = []
        last_dates = forecast.tail(request.days)
        for _, row in last_dates.iterrows():
            predictions.append({
                "data": row["ds"].strftime("%Y-%m-%d"),
                "previsao_fechamento": round(row["yhat"], 2)
            })
            
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Endpoint de verificação de saúde da API
    Retorna:
        dict: Status dos componentes principais
    """
    status = {
        "api": "healthy",
        "model": "loaded" if os.path.exists(MODEL_PATH) else "missing",
        "last_trained": datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat() \
                      if os.path.exists(MODEL_PATH) else "unknown",
        "version": "1.0.0"
    }
    return status
