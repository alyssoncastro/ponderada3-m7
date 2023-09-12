# ponderada3-m7
Atividade 3: Deploy de modelo de Machine Learning na Nuvem

### Autor: Alysson Cordeiro

#### agradecimento a mim mesmo. Primeira ativdade sozinho. Contudo, com muitos erros. Surgiro que leia a documentação completa para entender dos problemas gerado.

### Pré-requisitos.

Antes de prosseguir com a execução do projeto, é importante garantir que você tenha as seguintes bibliotecas e ferramentas corretamente configuradas em seu ambiente:

1. Python 3.6 ou superior (para o pycaret)
2. Pandas
3. PyCaret
4. FastAPI
5. Uvicorn
6. Pydantic
7. Unidecode

Você pode usar o `!pip` para instalar essas bibliotecas conforme necessário para garantir que seu ambiente esteja pronto para a execução do projeto.

###ESTRUTURAÇÃO

O projeto se divide em duas partes:  no vs code pré-processamento (devido à facilidade para limpar os dados) e um Notebook no Colab.

Notebook no Colab [aula1-semana6-pycaret.ipynb](https://colab.research.google.com/drive/1hIOZxtt_yAt7XHAQvt2oQ1rcdQcktI_S#scrollTo=5-gkNDY_MHyT).

- O Notebook no Colab contém o código utilizado já pré-processado do dados e o treinamento do modelo PyCaret.
- O modelo treinado é salvo e posteriormente carregado pelo script Python no vs code.
- Dentro do PyCaret, o modelo com a melhor acurácia foi o **Gradient Boosting Classifier**.
  
## Tentando executar a Api

etapas:

Abra um terminal e navegue até o diretório onde o arquivo "api.py" está localizado.

Execute o seguinte comando para iniciar o servidor FastAPI:

```python
uvicorn api:app --reload
```

A partir daqui não será possível acessar devido ao erro que expliquei a longo do documento.


