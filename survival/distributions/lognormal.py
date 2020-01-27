import numpy as np
from scipy.stats import norm
from survival.misc.sigmoid import *
from survival.distributions.basemodel import *


class Lognormal(Base):
    def __init__(self, mu=0, sigma=0.5, ti=None, xi=None):
        if ti is not None:
            self.train_org = ti
            self.train_inorg = xi
            self.gradient_descent()
            self.k = self.mu
            self.lmb = self.sigma
        else:
            self.train = []
            self.test = []
            self.train_org = []
            self.train_inorg = []
            self.mu = mu
            self.sigma = sigma
            self.params = []

    def determine_params(self, k, lmb, params):
        return super(Lognormal, self).determine_params(k, lmb, params)

    def pdf(self, t, mu=None, sigma=None, params=None):
        [mu, sigma] = self.determine_params(mu, sigma, params)
        return 1 / ((2 * np.pi)**.5 * sigma * t) * np.exp(-(np.log(t) - mu)**2 / (2 * sigma**2))

    def cdf(self, t, mu=None, sigma=None, params=None):
        [mu, sigma] = self.determine_params(mu, sigma, params)
        return norm.cdf((np.log(t) - mu) / sigma)

    def survival(self, t, mu=None, sigma=None, params=None):
        [mu, sigma] = self.determine_params(mu, sigma, params)
        return 1 - self.cdf(t, mu, sigma)

    def logpdf(self, t, mu, sigma):
        return -np.log(2 * np.pi) / 2 - np.log(sigma) - np.log(t) - ((np.log(t) - mu)**2 / (2 * sigma**2))

    def logsurvival(self, t, mu, sigma):
        return np.log(self.survival(t, mu, sigma))

    def loglik(self, t, x, mu, sigma):
        return sum(self.logpdf(t, mu, sigma)) + sum(self.logsurvival(x, mu, sigma))

    def grad(self, t, x, mu, sigma):
        n = len(t)
        m = len(x)
        z = (np.log(x) - mu) / sigma
        delmu = sum(np.log(t) - mu) / sigma**2 + \
            sum(norm.pdf(z) / norm.cdf(-z)) / sigma
        delsigma = -n * 1.0 / sigma + \
            sum((np.log(t) - mu)**2) / sigma**3 + \
            sum(z * norm.pdf(z) / norm.cdf(-z)) / sigma
        return np.array([delmu, delsigma])

    def numerical_grad(self, t, x, mu, sigma):
        eps = 1e-5
        delk = (self.loglik(t, x, mu + eps, sigma) -
                self.loglik(t, x, mu - eps, sigma)) / 2 / eps
        dellmb = (self.loglik(t, x, mu, sigma + eps) -
                  self.loglik(t, x, mu, sigma - eps)) / 2 / eps
        return np.array([delk, dellmb])

    '''
    def gradient_descent(self, numIter=2001, params = np.array([1.0,1.0])):
        for i in range(numIter):
            #lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1],params[2])
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1])
            params2 = params + 1e-9*directn
            lik = self.loglik(self.train_org,self.train_inorg,params2[0],params2[1])
            for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1]:
                params1 = params + alp1 * directn
                if(min(params1) > 0):
                    lik1 = self.loglik(self.train_org,self.train_inorg,params1[0],params1[1])
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
            params = params2
            if i%100 == 0:
                print("Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn))
                print("\n########\n")
        [self.mu,self.sigma] = params
        self.params = params
        #return params
    '''


#[1] http://home.iitk.ac.in/~kundu/paper160.pdf
