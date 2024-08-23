# História:
# Correlacionar 'aprovação' em uma matéria com 'presença', 'fazer-exercicio', 'le-livro'

import numpy as np
from typing import Literal

class BaseDados:
    def __init__(self) -> None:
        pass

    def gerar(quantidade : int) -> np.ndarray :
        dados = np.empty(shape=(quantidade, 2))

        for i in range(quantidade):
            r1 = np.random.rand()

            presente : Literal[0, 1] = 1 if np.random.rand() >= 0.5 else 0
            aprovado = 1 if (presente == 1 and r1 < 0.7) or (presente == 0 and r1 < 0.3)  else 0

            dados[i] = (presente, aprovado)
        
        return dados