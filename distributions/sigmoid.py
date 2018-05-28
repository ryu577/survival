import numpy as np

class Sigmoid():
    def __init__(self,c):
        self.upper_bound = c

    def transformed(self,x):
        return self.upper_bound/(1+np.exp(-x))

    def grad(self,x):
        p = self.transformed(x)/self.upper_bound
        return self.upper_bound*p*(1-p)

