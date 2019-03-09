import numpy as np
import unittest
from distributions.weibull import Weibull
from distributions.lomax import Lomax
from distributions.loglogistic import LogLogistic
from distributions.expmixture import ExpMix
from optimization.optimizn import bisection
import time


class TestDistributions(unittest.TestCase):
    def tst_weibull(self):
        self.assertTrue(tst_weibull())

    def tst_lomax(self):
        self.assertTrue(tst_lomax())

    def tst_loglogistic(self):
        self.assertTrue(tst_loglogistic())
    
    def tst_expmix_em(self):
        self.assertTrue(tst_expmix_em())


def tst_weibull():
    t = Weibull.samples_(1.1, 200, size=10000)
    start = time.time()
    params = Weibull.est_params(t)
    end = time.time()
    print("Estimating parameters of Weibull took: " + str(end-start))
    return abs(params[0]-1.1) < 1e-2

def tst_lomax():
    t = Lomax.samples_(1.1, 50, size=10000)
    start = time.time()
    params = Lomax.est_params(t)
    end = time.time()
    print("Estimating parameters of Weibull took: " + str(end-start))
    return abs(params[0]-1.1) < 1e-1

def tst_loglogistic():
    t = LogLogistic.samples_(10, 1.2, size=10000)
    start = time.time()
    ll = LogLogistic(ti=t, xi=np.array([]))
    end = time.time()
    print("Estimating parameters of LogLogistic took: " + str(end-start))
    return abs(ll.alpha-1.1) < 1e-2

def tst_expmix_em(mu_o=1/10, lmb_o=1/5, u_o=0.8, c=8):
    s, t, x, xs, xt = ExpMix.samples_(mu_o,lmb_o,u_o,50000,c)
    em = ExpMix(s, t, x, xs, xt)
    em.estimate_em(verbose=True)
    return abs(em.mu-mu_o) < 1e-2

def tst_expmix_em_raw(mu_o=1/10, lmb_o=1/5, u_o=0.8, c=8):
    s, t, x, xs, xt = ExpMix.samples_(mu_o,lmb_o,u_o,50000,c)
    ns=len(s); nt=len(t); nx=len(x)
    lmb=len(t)/sum(t); mu=len(s)/sum(s)
    for tt in range(500):
        lmb_sur = np.mean(np.exp(-lmb*xt))
        mu_sur = np.mean(np.exp(-mu*xs))
        u = ns*(1-lmb_sur)/(ns*(1-lmb_sur)+nt*(1-mu_sur))
        ## The actual probability of seeing sample beyong censor pt from sample-1.
        tau = u*np.exp(-mu*x)/(u*np.exp(-mu*x)+(1-u)*np.exp(-lmb*x))
        ## Use tau to estimate the rate parameters.
        mu = len(s)/(sum(s)+sum(tau*x))
        lmb = len(t)/(sum(t)+sum((1-tau)*x))
        if tt%100 == 0:
            print("mu:" + str(mu) + ", lmb:"+str(lmb)+", u:"+str(u))


sampl1 = LogLogistic.samples_(10,1.1)
sampl2 = LogLogistic.samples_(8,0.8)

ti = np.concatenate((sampl1,sampl2),axis=0)
xi = np.array([.1])


ll= LogLogistic(ti=ti, xi=xi, verbose=True)

start = time.time()
ll= LogLogistic(ti=ti, xi=xi, verbose=True)
end = time.time()
print("Estimating parameters of Weibull took: " + str(end-start))

