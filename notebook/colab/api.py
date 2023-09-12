import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib

# Carregando o modelo treinado
modelo = joblib.load('')


app = FastAPI()


class DadosEntrada(BaseModel):
    feature1: float
    feature2: float


# Definindo um endpoint
@app.post("/prever/")
async def prever(dados: DadosEntrada):
    dados_para_previsao = [[dados.feature1, dados.feature2]]

    previsao = modelo.predict(dados_para_previsao)[0]

    # Retornando a previsão
    return {"previsao": previsao}