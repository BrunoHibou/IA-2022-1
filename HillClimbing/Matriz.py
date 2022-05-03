import numpy as np


class Matrix:
    def __init__(self, rows, columns, max): #construtor da classe matriz, cria uma matriz aleatória
        self.matrix = np.random.randint(max, size = (rows, columns))

    def get_matrix(self):
        return self.matrix

if(__name__ == 'main'):
    a = Matrix(10,10,255)
    print(a.matrix)
    