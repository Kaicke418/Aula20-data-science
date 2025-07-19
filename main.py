import numpy as np
from sklearn.linear_model import LinearRegression



vendas = {
    'jan': 2000,
    'fev': 3000,
    'mar': 4000
}

meses = np.array([1, 2, 3]).reshape(-1, 1)
valores = np.array([2000, 3000, 4000])

modelo = LinearRegression()

modelo.fit(meses, valores)

proximo_mes = 4
venda_prevista = modelo.predict([[proximo_mes]])[0]

print(f'preisão de venda para o próximo mês: - {proximo_mes} -> {venda_prevista}')

