import requests
import json

# Testando o endpoint de previsão
resposta = requests.post("http://localhost:8000/predict", json={"days": 7})
print(json.dumps(resposta.json(), indent=2))
