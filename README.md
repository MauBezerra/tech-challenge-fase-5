# Stock Price Prediction with Facebook Prophet

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Sistema de previsão de preços de ações usando o modelo Prophet do Facebook com dados em tempo real da API Alpha Vantage.

## ✨ Funcionalidades

- **Coleta de dados em tempo real** da API Alpha Vantage
- **Previsão de séries temporais** com Facebook Prophet
- **API REST** para consulta de previsões
- **Arquitetura modular** para fácil manutenção
- **Configuração por ambiente** com arquivos `.env`

## 📂 Estrutura do Projeto

```bash
stock_prediction/
├── data/                  # Dados históricos de ações (CSV)
├── model/                 # Modelos Prophet treinados
├── src/                   # Código fonte
│   ├── data/              # Scripts de coleta de dados
│   ├── ml/                # Treinamento do modelo
│   └── api/               # API de previsões
├── notebooks/             # Jupyter notebooks para análise
├── tests/                 # Testes automatizados
├── .env.example           # Template de configuração
├── requirements.txt       # Dependências Python
└── README.md              # Documentação do projeto
```

## 🚀 Começando

### Pré-requisitos
- Python 3.8+
- Chave de API Alpha Vantage (disponível gratuitamente)

### Instalação
1. Clone o repositório
```bash
git clone https://github.com/seuusuario/stock_prediction.git
cd stock_prediction
```

2. Crie e ative o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências
```bash
pip install -r requirements.txt
```

4. Configure o ambiente
```bash
cp .env.example .env
# Edite o .env com sua chave da API Alpha Vantage
```

## 🛠️ Como Usar

1. **Coletar dados de ações**
```bash
python src/data/download_alphavantage.py
```

2. **Treinar modelo de previsão**
```bash
python src/ml/train_model.py
```

3. **Executar API de previsões**
```bash
python src/api/teste_api.py
```

## 📈 Exemplo de Previsão
```json
[
  {
    "data": "20/06/2025",
    "previsao_fechamento": 156.42
  },
  {
    "data": "21/06/2025",
    "previsao_fechamento": 157.15
  }
]
```

## 🌐 API de Previsão

### Iniciar a API
```bash
uvicorn src.api.main:app --reload
```

### Endpoints
- `POST /predict` - Recebe número de dias e retorna previsões

### Exemplo de Uso
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"days": 5}
)

if response.status_code == 200:
    for pred in response.json():
        print(f"Data: {pred['data']} | Previsão: ${pred['previsao_fechamento']:.2f}")
else:
    print(f"Erro: {response.json()['detail']}")
```

Acesse `http://localhost:8000/docs` para documentação interativa.

## 🤝 Como Contribuir
Contribuições são bem-vindas! Abra uma issue ou envie um pull request.

## 📜 Licença
Distribuído sob licença MIT. Veja [LICENSE](LICENSE) para mais informações.

## 📧 Contato
Seu Nome - seu.email@exemplo.com  
Link do Projeto: [https://github.com/seuusuario/stock_prediction](https://github.com/seuusuario/stock_prediction)

## Stock Prediction - Procter & Gamble (PG) Model

## 📊 Performance Metrics
- **Validation MAPE**: 3.27%
- **Test MAPE**: 5.65%
- **RMSE**: $7.77
- **R²**: -4.64 (indicates high market volatility captured)

## 🛠 Model Specifications
```python
{
  'changepoint_prior_scale': 0.03,
  'seasonality_mode': 'multiplicative',
  'regressors': ['bias_adjust'],
  'special_events': ['earnings_pg', 'us_holidays']
}
```

## 📅 Event Calendar
Key dates affecting predictions:
- **Earnings Dates**: 20 Jan, 21 Apr, 28 Jul, 20 Oct (±3 days)
- **Major Holidays**: July 4th, Thanksgiving, Christmas

## 📈 How to Use
```python
# Load model
import joblib
model = joblib.load('model/prophet_model.joblib')

# Make predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

![Validation Plot](model/validation_plot.png)

## 📊 Monitoramento

O sistema gera logs detalhados em `model_training.log` com:
- Tempo de execução
- Métricas de performance
- Erros ocorridos

## 🧪 Testes Automatizados

Para executar os testes:
```bash
pytest tests/
```

## 🔍 Exemplo de Request/Response

**Request**:
```json
POST /predict
{
  "days": 7
}
```

**Response**:
```json
[
  {
    "data": "2023-06-19",
    "previsao_fechamento": 145.32
  },
  ...
]
```

## 🚨 Tratamento de Erros

Códigos de erro comuns:
- `400`: Parâmetros inválidos
- `500`: Erro interno do modelo
