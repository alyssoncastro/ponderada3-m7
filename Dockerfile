# Use a imagem base oficial do Python
FROM python:3.8-slim

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .
COPY modelo_2.pkl .
COPY api.py .

# Instale as dependências do aplicativo
RUN pip install -r requirements.txt

# Copie todo o conteúdo do diretório local para o diretório de trabalho no contêiner
COPY . .

# Exponha a porta em que o aplicativo FastAPI estará em execução
EXPOSE 8000

# Comando para iniciar o aplicativo FastAPI
# no lugar do app deveria ser meu arquivo api em pkl
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
