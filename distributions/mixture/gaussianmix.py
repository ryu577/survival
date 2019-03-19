import numpy as np
from scipy.stats import norm

class GaussMix():    
    def __init__(self,mu1,sigma1,mu2,sigma2,p):
        self.mu1=mu1; self.sigma1=sigma1; self.mu2=mu2
        self.sigma2=sigma2; self.p=p
    
    @staticmethod
    def samples_(mu1, sigma1, mu2, sigma2, p, n_samples=5000):
        sampl_1_len = int(n_samples*p)
        sampl_2_len = int(n_samples*(1-p))
        x1 = np.random.normal(mu1,sigma1,sampl_1_len)
        x2 = np.random.normal(mu2,sigma2,sampl_2_len)
        x = np.concatenate((x1,x2),axis=0)
        return x

    def samples(self,n_samples=5000):
        mu1,mu2,sigma1,sigma2,p=self.mu1,self.mu2,self.sigma1,self.sigma2,self.p
        return GaussMix.samples_(mu1,mu2,sigma1,sigma2,p,n_samples)

    @staticmethod
    def loglik_(x, mu1, sigma1, mu2, sigma2, p):
        return sum(norm.pdf(x,mu1,sigma1)*p+norm.pdf(x,mu2,sigma2)*(1-p))

    def loglik(self,x):
        mu1,mu2,sigma1,sigma2,p=self.mu1,self.mu2,self.sigma1,self.sigma2,self.p
        return GaussMix.loglik_(x,mu1,mu2,sigma1,sigma2,p)

    @staticmethod
    def grad_(x,mu1, sigma1, mu2, sigma2, p):
        delp_numr = norm.pdf(x,mu1,sigma1)-norm.pdf(x,mu2,sigma2)
        delp_denom = norm.pdf(x,mu1,sigma1)*p+norm.pdf(x,mu2,sigma2)*(1-p)
        delp_trms = delp_numr/delp_denom
        delp = sum(delp_numr/delp_denom)
        delmu1_numr = p*(x-mu1)/sigma1**2*norm.pdf(x,mu1,sigma1)
        delmu1_trms = delmu1_numr/delp_denom
        delmu1 = sum(delmu1_trms)
        delmu2_numr = (1-p)*(x-mu2)/sigma2**2*norm.pdf(x,mu2,sigma2)
        delmu2_trms = delmu2_numr/delp_denom
        delmu2 = sum(delmu2_trms)
        delsigma1_numr = p*norm.pdf(x,mu1,sigma1)*((x-mu1)**2/sigma1**3-1/sigma1)
        delsigma1_trms = delsigma1_numr/delp_denom
        delsigma1 = sum(delsigma1_trms)
        delsigma2_numr = (1-p)*norm.pdf(x,mu2,sigma2)*((x-mu2)**2/sigma2**3-1/sigma2)
        delsigma2_trms = delsigma2_numr/delp_denom
        delsigma2 = sum(delsigma2_trms)
        return np.array([delmu1,delsigma1,delmu2,delsigma2,delp])

    def grad(self, x):
        mu1,mu2,sigma1,sigma2,p=self.mu1,self.mu2,self.sigma1,self.sigma2,self.p
        return GaussMix.grad_(x,mu1,mu2,sigma1,sigma2,p)

    @staticmethod
    def numr_grad_(x,mu1,sigma1,mu2,sigma2,p):
        eps = 1e-5
        prms = np.array([mu1,sigma1,mu2,sigma2,p])
        grd = np.zeros(5)
        for i in range(5):
            prms[i] += eps
            [mu1,sigma1,mu2,sigma2,p]=prms
            lik1 = GaussMix.loglik_(x,mu1,sigma1,mu2,sigma2,p)
            prms[i] -= 2*eps
            [mu1,sigma1,mu2,sigma2,p]=prms
            lik2 = GaussMix.loglik_(x,mu1,sigma1,mu2,sigma2,p)
            grd[i] = (lik2-lik1)/2/eps
            prms[i] += eps
        return grd

    def numr_grad(self,x):
        mu1,mu2,sigma1,sigma2,p=self.mu1,self.mu2,self.sigma1,self.sigma2,self.p
        return GaussMix.numr_grad_(x,mu1,mu2,sigma1,sigma2,p)


