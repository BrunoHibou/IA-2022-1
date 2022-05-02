import numpy as np

class matrix:
    def __init__(self, rows, columns, max): #construtor da classe matriz, cria uma matriz aleatória
        matrix = np.random.randint(max, size = (rows, columns))
        print(matrix)
        return matrix

if __name__ == "main":
    a = matrix()
    a.generate(10,10)
    