"""
Streamlit UI para previsão do preço da ação PG usando modelo Prophet já treinado.

Execute localmente com:
    streamlit run streamlit_app.py
Ou faça deploy na Streamlit Community Cloud para gerar um link público.
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date, timedelta

# Diretório que contém os arquivos *.joblib* gerados no treinamento
MODELS_DIR = Path(__file__).parent / "model"

@st.cache_resource(show_spinner="Carregando modelo...")
def load_model(ticker: str):
    """Carrega o modelo Prophet referente ao *ticker* escolhido.
    Caso não exista um modelo específico, utiliza o modelo padrão (PG)."""
    path = MODELS_DIR / f"prophet_{ticker}.joblib"
    if not path.exists():
        path = MODELS_DIR / "prophet_model.joblib"  # modelo padrão
    return joblib.load(path)

st.title("📈 Previsão de Preço de Ações com Prophet")

st.markdown("Selecione o ticker e o intervalo de datas para obter a previsão do preço de fechamento.")

# Entradas do usuário
# Lista de tickers disponíveis – adicione mais conforme modelos forem treinados
TICKERS = ["PG"]

ticker = st.selectbox("Ticker da ação", options=TICKERS, index=0)

date_range = st.date_input(
    "Período para previsão (início e fim)",
    value=(date.today(), date.today()),
)
# Trata todas as possibilidades de retorno do st.date_input
if isinstance(date_range, (tuple, list)):
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:  # apenas um elemento
        start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

event_flag = st.checkbox("Evento corporativo relevante?", value=False)

if st.button("🔮 Prever"):
    model = load_model(ticker)

    # Construir dataframe para todo o intervalo selecionado
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df_pred = pd.DataFrame(
        {
            "ds": dates,
            "bias_adjust": 1.0,  # regressora fixa
            "Volume": 1_000_000,
            "event_peak": int(event_flag),
        }
    )

    forecast = model.predict(df_pred)

    # Exibe gráfico de linha com a série prevista
    st.line_chart(forecast.set_index("ds")[["yhat"]], height=300)

    # Tabela detalhada
    with st.expander("Detalhes da previsão"):
        st.dataframe(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            .set_index("ds")
            .style.format({"yhat": "${:,.2f}", "yhat_lower": "${:,.2f}", "yhat_upper": "${:,.2f}"})
        )
