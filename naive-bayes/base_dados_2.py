# História:
# Correlacionar 'aprovação' em uma matéria com 'presença', 'fazer-exercicio', 'le-livro'

import numpy as np
from numpy.dtypes import StrDType
from sklearn.preprocessing import LabelEncoder

class BaseDados:
    presenteEncoder = LabelEncoder()
    aprovadoEncoder = LabelEncoder()

    def __init__(self) -> None:
        pass

    def gerar(self, quantidade : int) -> np.ndarray :
        dados = np.ndarray(shape=(quantidade, 2), dtype=StrDType)

        for i in range(quantidade):
            r1 = np.random.rand()

            presente = 'Presente' if np.random.rand() >= 0.5 else 'Ausente'
            aprovado = 'Aprovado' if (presente == 'Presente' and r1 < 0.7) or (presente == 'Ausente' and r1 < 0.8)  else 'Reprovado'

            dados[i] = (presente, aprovado)

        self.presenteEncoder = self.presenteEncoder.fit( dados[:, 0] )
        self.aprovadoEncoder = self.aprovadoEncoder.fit( dados[:, 1] )

        return dados
    
    def encode(self, dados: np.ndarray):
        x = dados[:, 0]
        y = dados[:, 1]

        presente = self.presenteEncoder.transform(dados[:, 0])
        aprovado = self.aprovadoEncoder.transform(dados[:, 1])

        x = np.empty(shape=( dados.shape[0], dados.shape[1] - 1,))

        x[:, 0] = presente
        y = aprovado

        return x, y
    
    def encodeX(self, xCategorico: np.ndarray):
        x = np.empty(shape=xCategorico.shape)
        
        x[:, 0] = self.presenteEncoder.transform(xCategorico[:, 0])

        return x

    def decodeY(self, y: np.ndarray):
        return self.aprovadoEncoder.inverse_transform(y)