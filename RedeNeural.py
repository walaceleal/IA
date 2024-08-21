import time
import numpy as np

class RedeNeural:
    def __init__(self):
        self.w = np.array([0.5])
        self.b = 0.5
        self.taxaAprendizado = 0.00000001

    def calcularCusto(self, Yp: np.ndarray, Y: np.ndarray):
        erros = 0.5 * pow(Yp - Y, 2) / Yp.shape[0]
        return np.sum(erros, axis=0, keepdims=True)

    def passoTreinamento(self, x: np.ndarray, Yp: np.ndarray, Y: np.ndarray, i):
        self.w -= self.taxaAprendizado * i * np.sum((Yp - Y) * x, axis=0)
        self.b -= self.taxaAprendizado * i * np.sum((Yp - Y), axis=0)

    def treinar(self, x: np.ndarray, y: np.ndarray):
        avgY = 1 #np.average(y)
        y = y / avgY

        avgX = 1 #np.average(x)
        x = x / avgX

        print(f'{avgY=} {avgX=}')

        custo = np.array([9999]) 
        passos = 1

        while( passos <= 2000):
            Yp = self.prever(x)
            self.passoTreinamento(x=x, Y=y, Yp=Yp, i=1)
            custo = self.calcularCusto(Yp, y)
            passos += 1
            print(f'({self.w[0]}, {self.b}) {custo=} {passos=}')
            #time.sleep(0.5)
            
    
        print(f'{self.w=} {self.b}')     
        
        self.w *= avgY/avgX
        self.b *= avgY

        print(f'{self.w=} {self.b} {avgY=} {avgX=}')            

        

        #print(f'{X=}')
        #print(f'{Y=}')
        #print(f'{Yp=}')
        #print(f'{erros=}')
        #print(f'{custo=}')

    def derivada(self, X):
        return X

    def prever(self, X: np.ndarray):
        return X * self.w.T + self.b