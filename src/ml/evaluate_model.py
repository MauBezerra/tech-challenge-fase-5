"""
Avaliação de um modelo Prophet previamente treinado e salvo.

Etapas executadas:
1. Carrega dados históricos e o arquivo `prophet_model.joblib` salvo durante o treinamento.
2. Converte dataframe para formato Prophet e define recorte temporal (`TEST_SIZE`) para hold-out.
3. Recria colunas regressoras (`bias_adjust`, `event_peak`, `Volume`) idênticas às do treinamento.
4. Caso o objeto Prophet carregado não contenha histórico (`history`), re-treina rapidamente sobre o dataset de treino para habilitar previsão.
5. Gera previsões para o horizonte de teste (`TEST_SIZE`).
6. Calcula métricas quantitativas: MAPE, RMSE e R².
7. Cria explicações qualitativas baseadas nos valores das métricas para auxiliar interpretação não-técnica.
8. Persiste dicionário de métricas em `model/metrics.json` e imprime resumo no console.
"""
import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# Configurações
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(BASE_DIR, 'data', 'stock_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'prophet_model.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'model', 'metrics.json')

TEST_SIZE = 30  # dias finais


def main():
    # Carregar dados e modelo
    data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    df = data.reset_index().rename(columns={'index': 'ds', 'Close': 'y'})

    train_df = df.iloc[:-TEST_SIZE]
    test_df = df.iloc[-TEST_SIZE:]

    # Garante colunas regressoras obrigatórias
    train_df['bias_adjust'] = 1.0
    test_df['bias_adjust'] = 1.0

    # Criar feature event_peak com mesma lógica do treino
    THRESH_PCT = 0.03
    THRESH_VOL = data['Volume'].quantile(0.99)
    train_df['event_peak'] = ((train_df['y'].pct_change().abs() > THRESH_PCT) | (train_df['Volume'] > THRESH_VOL)).astype(int)
    test_df['event_peak'] = ((test_df['y'].pct_change().abs() > THRESH_PCT) | (test_df['Volume'] > THRESH_VOL)).astype(int)

    # Volume deve existir como float
    train_df['Volume'] = train_df['Volume'].astype(float)
    test_df['Volume'] = test_df['Volume'].astype(float)

    model = joblib.load(MODEL_PATH)

    # Re-treinar rápido se necessário (Prophet requer ajuste completo)
    if not hasattr(model, 'history'):  # caso modelo salvo não tenha atributo
        model.fit(train_df)

    future = model.make_future_dataframe(periods=TEST_SIZE)
    future['bias_adjust'] = 1.0
    volume_mean = train_df['Volume'].rolling(5).mean().iloc[-1]
    future['Volume'] = volume_mean
    future['event_peak'] = 0
    forecast = model.predict(future).tail(TEST_SIZE)

    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values

    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    # Gerar explicações simplificadas
    def mape_qual(m):
        return 'ótimo' if m < 5 else 'bom' if m < 10 else 'ruim'
    def r2_qual(r):
        return 'ótimo' if r > 0.5 else 'aceitável' if r > 0 else 'ruim'

    metrics = {
        'timestamp': datetime.utcnow().isoformat(),
        'test_size': TEST_SIZE,
        'mape': round(mape, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'explanations': {
            'mape': f"MAPE (erro percentual médio) mostra que em média as previsões estão {round(mape,2)}% distantes do real — quanto menor melhor; abaixo de 5% é considerado {mape_qual(mape)}.",
            'rmse': "RMSE mede o erro médio absoluto em dólares; números menores indicam previsões mais próximas.",
            'r2': f"R² indica quanta variabilidade o modelo explica; valores perto de 1 são bons. O valor atual ({round(r2,2)}) é considerado {r2_qual(r2)}."
        }
    }

    # Salvar métricas
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print('Metrics saved to', METRICS_PATH)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
