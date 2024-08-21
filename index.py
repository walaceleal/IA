import numpy as np
from RedeNeural import RedeNeural

N = 200

Y = np.empty(shape=(2*N)) #np.linspace(start=-0.10, stop=0.40, num=N) - 0.000001 * np.random.rand((N)) + 0.000001 * np.random.rand((N))
X = np.empty(shape=(2*N))

for i in range(0, 2*N):
    Y[i] = -13 * (i-N) + 31
    X[i] = (i-N)

R1 = RedeNeural()

R1.treinar(X, Y)

Yp = R1.prever(
    np.array([10, 100, 1000])
)

print(Yp)
