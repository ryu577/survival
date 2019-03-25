import numpy as np
from distributions.mixture.basemixture import *

class ExpMix(BaseMix):
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
        return sum(np.log(u*mu*np.exp(-mu*x)+(1-u)*lmb*np.exp(-lmb*x)))

    def loglik(self, x):
        return ExpMix.loglik_(x, self.mu, self.lmb, self.u)

    def loglik_prms(self, prms):
        [mu, lmb, u] = prms
        return ExpMix.loglik_(self.x, mu,lmb,u)

    @staticmethod
    def sample_(mu, lmb, u, nsamples=5000):
        n1 = int(nsamples*u)
        n2 = int(nsamples*(1-u))
        smpl1 = np.random.exponential(1/mu, size=n1)
        smpl2 = np.random.exponential(1/lmb, size=n2)
        return np.concatenate((smpl1,smpl2),axis=0)

