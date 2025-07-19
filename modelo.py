from sklearn.linear_model import LinearRegression
import numpy as np
dados = {
    'JULHO': 500000,

    'AGOSTO': 700000,

    'SETEMBRO': 900000,

    'OUTUBRO': 90000,

    'NOVEMBRO': 1000000,

    'DEZEMBRO': 200000}

meses = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
vendas = np.array([500000, 700000, 900000, 90000, 1000000, 200000])

modelo = LinearRegression()
modelo.fit(meses, vendas)

print(meses, vendas)

proximo_mes = 7
venda_prevista = modelo.predict([[proximo_mes]])[0]
print(f'preisão de venda para o próximo mês: - {proximo_mes} -> {venda_prevista}')