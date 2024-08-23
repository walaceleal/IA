from base_dados_2 import BaseDados

import numpy as np
from sklearn.naive_bayes import GaussianNB

dados : np.ndarray = BaseDados.gerar(quantidade = 400)

x = dados[:,0].reshape((-1, 1))
y = dados[:,1].reshape((-1))


NB = GaussianNB()

NB.fit(X=x, y=y)

previsao = NB.predict([
    [0], 
    [1]
])

print(previsao)

