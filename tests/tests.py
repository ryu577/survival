import numpy as np
import unittest
from distributions.weibull import Weibull
from distributions.lomax import Lomax
from distributions.loglogistic import LogLogistic
from distributions.expmixture import ExpMix
from optimization.optimizn import bisection
import time
import matplotlib.pyplot as plt
from misc.misc import *
import pandas as pd


class TestDistributions(unittest.TestCase):
    def tst_weibull(self):
        self.assertTrue(tst_weibull())

    def tst_lomax(self):
        self.assertTrue(tst_lomax())

    def tst_loglogistic(self):
        self.assertTrue(tst_loglogistic())
    
    def tst_expmix_em(self):
        self.assertTrue(tst_expmix_em())

    def tst_expmix_em_weighted(self):
        self.assertTrue(tst_expmix_em_weighted())

    def tst_expct_dt_slopes(self):
        self.assertTrue(tst_expct_dt_slopes())


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
    print("Estimating parameters of Lomax took: " + str(end-start))
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

def tst_expmix_em_weighted(mu_o=1/10, lmb_o=1/5, u_o=0.8, c=8):
    """
    Weighing all vectors by the same amount shouldn't change the estimate.
    """
    s, t, x, xs, xt = ExpMix.samples_(mu_o,lmb_o,u_o,50000,c)
    ws = np.ones(len(s))*4; wt = np.ones(len(t))*4; wx = np.ones(len(x))*4
    em = ExpMix(s, t, x, xs, xt, ws, wt, wx)
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


def tst_expct_dt_slopes():
    sam = LogLogistic.samples_(1.1,10)
    ll = LogLogistic(sam,np.array([.01]))
    coefs = ll.expctd_downtime_linear_coeffs(1,3,10)
    xs = np.arange(.1,10.,.1)
    ys = ll.hazard(xs)
    plt.plot(xs, ys)
    opt_tau = get_opt_tau(ll.hazard,3.0)
    coefs2 = ll.expctd_downtime_linear_coeffs(opt_tau,opt_tau+15.0,3.0)
    return coefs2[0] >= 0


def predicn(train_df, x_features, trm="alpha"):
    y_alp = train_df[trm]
    X = x_features
    lhs = np.dot(X.T,X)
    rhs = np.dot(X.T,y_alp)
    betas = np.linalg.solve(lhs,rhs)
    y_pred = np.dot(X,betas)
    return y_pred, betas


def fast_loglogistic():
    train_df = pd.DataFrame()
    ##Training data set.
    for i in range(100):
        k = np.random.uniform()*2.0
        lmb = np.random.uniform()*20.0
        ti = LogLogistic.samples_(lmb,k)
        lmx = Lomax.est_params(ti)
        wbl = Weibull.est_params(ti)
        train_df = train_df.append({'alpha':lmb,'beta':k,\
                        'lomax_k':lmx[0],'lomax_lmb':lmx[1],
                        'weib_k':wbl[0],'weib_lmb':wbl[1]},
                        ignore_index=True)
    train_df["weib_lmb"][np.isinf(train_df["weib_lmb"])] = 1000.0
    train_df["weib_lmb"][train_df["weib_lmb"]>1000.0] = 1000.0
    train_df["lomax_k"][np.isnan(train_df["lomax_k"])]=10.0
    x_features = np.ones((len(train_df),15))
    x_features[:,1] = train_df["lomax_k"]
    x_features[:,2] = train_df["lomax_lmb"]
    x_features[:,3] = train_df["weib_k"]
    x_features[:,4] = train_df["weib_lmb"]
    x_features[:,5] = train_df["weib_k"]**2
    x_features[:,6] = train_df["weib_lmb"]**2
    x_features[:,7] = train_df["lomax_k"]**2
    x_features[:,8] = train_df["lomax_lmb"]**2
    x_features[:,9] = train_df["lomax_k"]*train_df["lomax_lmb"]
    x_features[:,10] = train_df["lomax_k"]*train_df["weib_k"]
    x_features[:,11] = train_df["lomax_k"]*train_df["weib_lmb"]
    x_features[:,12] = train_df["lomax_lmb"]*train_df["weib_k"]
    x_features[:,13] = train_df["lomax_lmb"]*train_df["weib_lmb"]
    x_features[:,14] = train_df["weib_k"]*train_df["weib_lmb"]
    y_alp, lin_alphas = predicn(train_df, x_features)
    y_beta, lin_betas = predicn(train_df, x_features, "beta")
    res_df = pd.DataFrame()
    for i in range(100):
        res_df = res_df.append({'actual_alpha':train_df["alpha"][i],'pred_alpha':y_alp[i],
                        'actual_beta':train_df["beta"][i], 'pred_beta':y_beta[i],
                        'weib_k':train_df["weib_k"][i], 'weib_lmb':train_df['weib_lmb'][i],
                        'lomax_k':train_df["lomax_k"][i], 'lomax_lmb':train_df['lomax_lmb'][i]},
                        ignore_index=True)
    bad_df = res_df[abs(res_df["actual_alpha"]-res_df["pred_alpha"])>3.0]
    good_df = res_df[abs(res_df["actual_alpha"]-res_df["pred_alpha"])<3.0]


def tst_lomax_weibull():
    sampl1 = LogLogistic.samples_(12.0,0.8)
    sampl2 = LogLogistic.samples_(8.0,1.1)
    ti = np.concatenate((sampl1,sampl2),axis=0)
    xi = np.array([.1])
    ll = LogLogistic(ti,xi)
    lmx = Lomax.est_params(ti)
    wbl = Weibull.est_params(ti)

