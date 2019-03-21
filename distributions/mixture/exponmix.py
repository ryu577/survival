import numpy as np

class ExpMix():
    def __init__(self, x, mu=None, lmb=None, u=None):
        if x is not None:
            self.x=x
            self.mu = self.lmb = 1/np.mean(x)
            self.u = 0.5
        if mu is not None:
            self.mu = mu; self.lmb=lmb; self.u=u
        self.params = np.array([self.mu, self.lmb, self.u])

    @staticmethod
    def loglik_(x, mu, lmb, u):
        return np.log(u*mu*np.exp(-mu*x)+(1-u)*lmb*np.exp(-lmb*x))

    def loglik(self, x):
        return ExpMix.loglik_(x, self.mu, self.lmb, self.u)

    
        
