import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

dados = pd.read_csv('dados.csv')

df = pd.DataFrame(dados)

X = np.array(df.index).reshape(-1, 1)

y= np.array(df['Tempo_espera'])

modelo = LinearRegression()

modelo.fit(X, y)

previsao = modelo.predict([[len(df) + 1]])

print(f'PREVISAO DE TEMPO DE ESPERA DO PRÓXIMO PACIENTE {previsao}')