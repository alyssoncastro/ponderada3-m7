FROM python:3.8-slim

# Configuração da pasta de trabalho no container
WORKDIR /app

# requisitos do projeto 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código da aplicação para o container
COPY . .

# Porta em que a aplicação Flask irá escutar
EXPOSE 5000

# Comando para iniciar a aplicação Flask
CMD ["python", "app.py"]
