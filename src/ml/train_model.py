"""
Treina um modelo Prophet para previsão de preços de ações.

Fluxo completo:
1. Configuração de logging e caminhos de dados/modelo.
2. Definição de feriados de mercado (US) e eventos corporativos (earnings PG) e união em `all_events`.
3. Definição de grade simples de hiper-parâmetros (`GRID`) e parâmetros padrão (`DEFAULT_PARAMS`).
4. Função `train_and_score` – recebe dados de treino/validação + hiper-parâmetros, treina modelo Prophet, gera previsões e retorna MAPE.
5. Bloco principal:
   5.1 Carrega `stock_data.csv` gerado pelo ETL (yfinance).
   5.2 Garante colunas auxiliares (`bias_adjust`) e cria feature `event_peak` baseada em picos de preço/volume.
   5.3 Prepara dataframe no formato Prophet (`ds`/`y`) e faz split temporal (últimos 60 dias para validação).
   5.4 Loop pela `GRID` seleciona melhores hiper-parâmetros via MAPE.
   5.5 Treina modelo final em todo o histórico com parâmetros ótimos + sazonalidades extras.
   5.6 Salva modelo com `joblib` em `model/prophet_model.joblib`.
   5.7 Gera gráfico de performance (últimos N dias) e salva em `model/`.
6. Mensagens de status são registradas em `model_training.log` e exibidas no console.
"""
import pandas as pd
import joblib
from prophet import Prophet
from prophet.plot import plot_plotly
import os
from pathlib import Path
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(BASE_DIR, 'data', 'stock_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'prophet_model.joblib')

# Feriados americanos
US_HOLIDAYS = [
    # 2023-2025
    '2023-01-01', '2023-01-16', '2023-02-20', '2023-04-07', '2023-05-29',
    '2023-06-19', '2023-07-04', '2023-09-04', '2023-11-11', '2023-11-23',
    '2023-12-25', '2024-01-01', '2024-01-15', '2024-02-19', '2024-04-29',
    '2024-05-27', '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-11',
    '2024-11-28', '2024-12-25', '2025-01-01', '2025-01-20', '2025-02-17',
    '2025-04-28', '2025-05-26', '2025-06-19', '2025-07-04'
]

us_holidays = pd.DataFrame({
    'holiday': 'us_market',
    'ds': pd.to_datetime(US_HOLIDAYS),
    'lower_window': -2,
    'upper_window': 1
})

"""
Eventos Corporativos da PG
"""
earnings_dates = pd.DataFrame({
    'holiday': 'earnings_pg',
    'ds': pd.to_datetime([
        '2023-01-20', '2023-04-21', '2023-07-28', '2023-10-20',
        '2024-01-19', '2024-04-19', '2024-07-26', '2024-10-18',
        '2025-01-24', '2025-04-18', '2025-07-25', '2025-10-17'
    ]),
    'lower_window': -3,
    'upper_window': 3
})

# Combinar com outros feriados
all_events = pd.concat([us_holidays, earnings_dates])

# Grade simples de hiperparâmetros
GRID = [
    {'changepoint_prior_scale': 0.03, 'seasonality_prior_scale': 0.1},
    {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 0.1},
    {'changepoint_prior_scale': 0.1,  'seasonality_prior_scale': 0.2},
    {'changepoint_prior_scale': 0.3,  'seasonality_prior_scale': 0.2}
]
DEFAULT_PARAMS = {
    'holidays_prior_scale': 0.3,
    'seasonality_mode': 'multiplicative',
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False,
    'holidays': all_events
}

def train_and_score(train_df, val_df, grid_params):
    """Treina Prophet com params e retorna MAPE na validação."""
    params = {**DEFAULT_PARAMS, **grid_params}
    m = Prophet(**params)
    m.add_regressor('bias_adjust', prior_scale=0.5)
    m.add_regressor('Volume', prior_scale=0.5)
    m.add_regressor('event_peak', prior_scale=0.5)
    # sazonalidades extras
    m.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=len(val_df))
    future['bias_adjust'] = 1.0
    volume_mean = train_df['Volume'].rolling(5).mean().iloc[-1]
    future['Volume'] = volume_mean
    future['event_peak'] = 0
    fcst = m.predict(future).tail(len(val_df))
    mape = (abs((val_df['y'].values - fcst['yhat'].values) / val_df['y'].values)).mean()*100
    return m, mape

try:
    logger.info(f"Iniciando treinamento para {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    """
    Preparação dos dados de treino
    """
    # Carregar dados
    data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # Garantir que bias_adjust existe
    data['bias_adjust'] = data.get('bias_adjust', 1.0)

    # Criar feature de pico de evento baseada em variação de preço/volume
    THRESH_PCT = 0.03  # 3%
    THRESH_VOL = data['Volume'].quantile(0.99)
    data['event_peak'] = (
        (data['Close'].pct_change().abs() > THRESH_PCT) |
        (data['Volume'] > THRESH_VOL)
    ).astype(int)

    # Preparar DataFrame para Prophet
    df = data.reset_index().rename(columns={'index': 'ds', 'Close': 'y'})

    # Split treino/validação (últimos 60 dias para validação)
    VAL_DAYS = 60
    train_df_part = df.iloc[:-VAL_DAYS].copy()
    val_df = df.iloc[-VAL_DAYS:].copy()

    # Regressor fixo bias_adjust
    train_df_part['bias_adjust'] = 1.0
    val_df['bias_adjust'] = 1.0

    best_mape = float('inf')
    best_params = None
    for grid_params in GRID:
        _, mape = train_and_score(train_df_part, val_df, grid_params)
        logger.info(f"Grid params {grid_params} => MAPE {mape:.2f}%")
        if mape < best_mape:
            best_mape = mape
            best_params = grid_params

    logger.info(f"Melhores params encontrados: {best_params} com MAPE={best_mape:.2f}%")

    # Treinar modelo final com melhores parâmetros em todo conjunto disponível
    # Força maior flexibilidade de tendência no modelo final
    final_params = {**DEFAULT_PARAMS, **best_params}
    model = Prophet(**final_params)
    model.add_regressor('bias_adjust', prior_scale=0.5)
    model.add_regressor('Volume', prior_scale=0.5)
    model.add_regressor('event_peak', prior_scale=0.5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df)


    print(f"\n Treinando modelo para: {data['Ativo'].iloc[0]}")
    print(f" Dados de: {df['ds'].min().date()} a {df['ds'].max().date()}")


    # Salvar modelo otimizado
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Modelo salvo em: {MODEL_PATH}")

    # Gerar gráfico de performance (últimos 180 dias)
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8')
        # Previsões in-sample
        forecast_full = model.predict(df)
        TAIL_DAYS = 30  # janela do gráfico de performance
        tail_days = TAIL_DAYS if len(df) > TAIL_DAYS else len(df)
        real_tail = df.tail(tail_days)
        pred_tail = forecast_full[['ds', 'yhat']].tail(tail_days)
        plt.figure(figsize=(10, 4))
        plt.plot(real_tail['ds'], real_tail['y'], label='Real')
        plt.plot(pred_tail['ds'], pred_tail['yhat'], label='Previsto')
        plt.title(f'Preço real vs. previsão (últimos {tail_days} dias)')
        plt.xlabel('Data'); plt.ylabel('Preço (USD)')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(BASE_DIR, 'model', f'performance_last{tail_days}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Gráfico de performance salvo em: {plot_path}")
    except Exception as plot_err:
        logger.warning(f"Falha ao gerar gráfico de performance: {plot_err}")

except Exception as e:
    logger.error(f"Erro durante o treinamento: {str(e)}")
    raise
finally:
    print("\n" + "═"*50)
    print(" Processo de treinamento concluído")
    print("═"*50)
