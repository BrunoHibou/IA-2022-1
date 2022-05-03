from Matriz import Matrix as matrix
import random as rand
import numpy as np

class HillClimbing:

    def __init__(self):
        aux = matrix(30, 30, 255)
        self.matrix = aux.get_matrix()
        self.neighbors = set()
        a = self.matrix.shape[0]-1
        b = self.matrix.shape[1]-1
        self.start = (rand.randint(0, a), rand.randint(0, b))
        self.last = None

    def compare(self, current, neighbor):
        
        self.neighbors.remove(neighbor)
        if(self.matrix[current[0]][current[1]] < self.matrix[neighbor[0]][neighbor[1]]):
            return neighbor
        else:

            return current
        
    def verify(self, position):
        if(0 <= position[0] < self.matrix.shape[0] and 0 <= position[1] < self.matrix.shape[1]):
            return True
        else:
            return False


    def HillClimb(self):
        
        current = self.start
        print('start:', self.start)
        print(self.matrix[self.start[0]][self.start[1]])
        new_current = current
        old_current = None
        counter = 1
        while True:   
            
            print(counter)
            counter = counter + 1 
            
            if(self.verify((new_current[0]+1, new_current[1]))):
                self.neighbors.add((new_current[0]+1, new_current[1]))
            
            if(self.verify((new_current[0]-1, new_current[1]))):
                self.neighbors.add((new_current[0]-1, new_current[1]))
            
            if(self.verify((new_current[0], new_current[1]+1))):
                self.neighbors.add((new_current[0], new_current[1]+1))
            
            if(self.verify((new_current[0], new_current[1]-1))):
                self.neighbors.add((new_current[0], new_current[1]-1))


            for neighbor in self.neighbors.copy():
                print(self.neighbors)
                aux = self.compare(new_current, neighbor)
                print('aux: ', aux)
                

                new_current = aux
                print('new: ', new_current)
                print(self.matrix[new_current[0]][new_current[1]])
                

            if old_current == new_current:
                self.last = new_current
                
                return self.matrix[new_current[0]][new_current[1]]
            old_current = new_current
            print('old', old_current)

climb = HillClimbing()
a = climb.HillClimb()
print(climb.matrix)


        