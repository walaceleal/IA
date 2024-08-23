from base_dados_2 import BaseDados

import numpy as np
from sklearn.naive_bayes import GaussianNB

# Busco da base de dados categoticos: Texto
BD = BaseDados()
dadosCategoricos : np.ndarray = BD.gerar(quantidade = 400)

# Transformo os dados categoricos em dados numéricos
xTreinamento, yTreinamento = BD.encode(dadosCategoricos)

# Treino o modelo Naive Bayes
NB = GaussianNB()
NB.fit(X=xTreinamento, y=yTreinamento)

# Antes de usar o modelo para prever, converto as entradas categóricas em numéricas.
xCategorico = np.array([
    ["Ausente"],
    ["Presente"]
], dtype=object)

x = BD.encodeX(xCategorico)
    
# Utilizo o modelo e converto para informação legível.
yCategorico = BD.decodeY(NB.predict(x))

previsao = np.ndarray(shape=( xCategorico.shape[0], xCategorico.shape[1] + 1 ), dtype=object)
previsao[:, 0:xCategorico.shape[1]] = xCategorico[:, :]
previsao[:, x.shape[1]] = yCategorico


print( previsao )