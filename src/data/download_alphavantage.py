"""
Script de download - Versão Internacional (substituição completa)
"""
import os
import argparse
import time
import pandas as pd
import requests
from pathlib import Path

# Configurações
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(BASE_DIR, 'data', 'stock_data.csv')

# Feriados americanos relevantes para PG
US_HOLIDAYS = [
    '2023-01-02', '2023-07-04', '2023-11-23', '2023-12-25',
    '2024-01-01', '2024-07-04', '2024-11-28', '2024-12-25',
    '2025-01-01', '2025-07-04', '2025-11-27', '2025-12-25'
]

# Configuração padrão para ações internacionais
def get_alpha_vantage_params(symbol):
    return {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',
        'datatype': 'csv'
    }

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--symbol', type=str, default='PETR4.SAO', help='Símbolo da ação')
parser.add_argument('--years', type=int, default=2, help='Anos de dados históricos')
args = parser.parse_args()

def get_alpha_vantage_data(symbol, years):
    """Coleta dados históricos da Alpha Vantage"""
    api_key = "D2UTNYK0LPHZOQVU"  # TEMPORÁRIO - remover depois
    
    print(f"\n📈 Coletando {years} ano(s) de dados para {symbol}...")
    
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    
    try:
        # Requisição com tratamento de limites
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"Resposta inesperada: {data.get('Note', 'Erro desconhecido')}")
        
        # Processar dados
        df = pd.DataFrame.from_dict(
            data['Time Series (Daily)'], 
            orient='index'
        ).sort_index()
        
        # Renomear colunas e filtrar período
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }).astype(float)
        
        df.index = pd.to_datetime(df.index)
        df = df.last(f'{years}Y')
        df['Ativo'] = symbol
        
        """
        Pré-processamento para PG
        """
        # Adicionar coluna de ajuste
        df['bias_adjust'] = 1  # Valor constante para correção de viés
        
        # Salvar dados processados
        df.to_csv(DATA_PATH, index=True)
        print(f"✅ Dados salvos em: {DATA_PATH} (substituição completa)")
        print(f"📅 Período: {df.index.min().date()} a {df.index.max().date()}")
        print(f"📊 Total de registros: {len(df)}")
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
    finally:
        # Respeitar limite da API
        time.sleep(12)  # 5 requests/minuto

if __name__ == "__main__":
    get_alpha_vantage_data(args.symbol, args.years)
