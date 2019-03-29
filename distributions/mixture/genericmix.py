import numpy as np

class GenericMix():
    """
    A generic mixture of two distributions.
    """
    
    def __init__(self,p,dist1,dist2,t=None):
        self.p = p
        self.dist1 = dist1
        self.dist2 = dist2
    
    def init_w_params(self, params):
        pass

    def init_w_data(self, t, w):
        pass

    def pdf(self,t):
        p = self.p
        return p*self.dist1.pdf(t)+(1-p)*self.dist2.pdf(t)
    
    def cdf(self,t):
        p = self.p
        return p*self.dist1.cdf(t)+(1-p)*self.dist2.cdf(t)

    @staticmethod
    def em(dist1, dist2):
        pass

