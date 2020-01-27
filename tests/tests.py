import numpy as np
import unittest
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.stats import nbinom

from survival.distributions.weibull import Weibull
from survival.distributions.lomax import Lomax
from survival.distributions.loglogistic import *
from survival.distributions.exponential import *
from survival.distributions.mixture.exponmix import *
from survival.distributions.mixture.exponmix_censored import *
from survival.distributions.mixture.gaussianmix import GaussMix
from survival.distributions.regressed.loglogisticregr import *
from survival.optimization.optimizn import bisection
from survival.misc.misc import *
import survival.ptprocesses.negativebinomial as nb


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

    def tst_loglik_w_features(self):
        self.assertTrue(tst_loglik_w_features())

    def tst_grad_w_features(self):
        self.assertTrue(tst_grad_w_features())

    def tst_expon_censor_full_info_loss(self):
        self.assertTrue(tst_expon_censor_full_info_loss())


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
    t = LogLogistic.samples_(1.2, 10, size=10000)
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
    xs = np.arange(.1,10.,.1)
    ys = ll.hazard(xs)
    plt.plot(xs, ys)
    opt_tau = get_opt_tau(ll.hazard,3.0)
    coefs2 = ll.expctd_downtime_linear_coeffs(opt_tau,opt_tau+15.0,3.0)
    return coefs2[0] >= 0


def tst_loglik_w_features():
    ti = np.array([1.0,1.0])
    xi = np.array([1.0,1.0])
    fsamples = np.array([[1,1],[1,1]])
    fcensored = np.array([[1,1],[1,1]])
    w = np.array([[1,1],[1,1]])
    ll = LogLogistic(ti,xi)
    ## Note that shape and scale upper bounds are 5 and 900 by default.
    loglik = LogLogisticRegr.loglikelihood_(ti, xi, fsamples, fcensored, ll, w)
    return abs(loglik + 55.83229) < 1e-3


def tst_grad_w_features():
    ti = np.array([1.0,1.0])
    xi = np.array([1.0,1.0])
    fsamples = np.array([[1,1],[1,1]])
    fcensored = np.array([[1,1],[1,1]])
    w = np.array([[1.0,1.0],[1.0,1.0]])
    ll = LogLogistic(ti,xi)
    grd = LogLogisticRegr.grad_(ti, xi, fsamples, fcensored, ll, w)
    grd_numr = LogLogisticRegr.numerical_grad_(ti, xi, fsamples, fcensored, ll, w)
    return abs(grd[0,0]-grd_numr[0,0]) < 1e-3


def tst_loglogistic_fitting():
    llr = LogLogisticRegr()
    w = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0]])
    llr.gradient_descent(w)


def tst_gaussmix_grad():
    gm = GaussMix(mu1=-2,sigma1=1,mu2=2,sigma2=1,p=0.3)
    x = gm.samples(1000000)
    grd = gm.numr_grad(x)
    return grd

def tst_exponmix_grad(size=500000):
    sams = ExpMix.sample_(.1,1.0,0.33,size)
    exp_mx = ExpMix(sams)
    return exp_mx.numr_grad_prms_(np.array([.1,1.0,.33]))/size

def tst_exponmix_censored_grad(size=100000):
    s,t,x_cen,xs,xt = CensrdExpMix.samples_(.7,1.0,0.33,size,1.1)
    cem = CensrdExpMix(s,t,x_cen,xs,xt)
    cem.loglik_prms(np.array([.7,1.0,.33]))
    prms = np.array([.7,1.0,.33])
    return cem.numr_grad_prms_(prms)/size

def tst_gaussmix_fit():
    gm = GaussMix(-2,1,2,1,0.3)
    x = gm.samples(1000)
    param = gm.numr_fit(x,2)
    return param


def tst_expon_censor_full_info_loss(size=10000):
    # Generate exponential data, censor it
    # and don't keep any record of the censored
    # data.
    lmb = 10.0
    all_dat = np.random.exponential(1/lmb,size=size)
    tau = 1/lmb
    t = all_dat[all_dat<tau]
    x = all_dat[all_dat>tau]
    lmb1 = Exponential.mle_censored_full_info_loss(t,tau)
    lmb2 = len(t)/(sum(t)+sum(x))
    return abs(lmb1-lmb)/lmb < 1e-1


def tst_exponmix_censored_simple_est(size=100000):
    mu=0.7; lmb=1.0; u=0.33; tau=1.1
    s,t,x_cen,xs,xt = CensrdExpMix.samples_(mu,lmb,u,size,tau)
    cem = CensrdExpMix(s,t,x_cen,xs,xt)
    mu_hat = Exponential.mle_censored_full_info_loss(s,tau)
    lmb_hat = Exponential.mle_censored_full_info_loss(t,tau)
    u = CensrdExpMix.u_from_lmb_mu_simplified(mu_hat, lmb_hat, s,t,tau)
    return mu_hat, lmb_hat, u


#######################################
##
#######################################

def fast_loglogistic():
    lin_alphas, lin_betas, y_alp, y_beta, train_df = LogLogistic.train_fast_()
    res_df = pd.DataFrame()
    for i in range(100):
        res_df = res_df.append({'actual_alpha':train_df["alpha"][i],'pred_alpha':y_alp[i],
                        'actual_beta':train_df["beta"][i], 'pred_beta':y_beta[i],
                        'weib_k':train_df["weib_k"][i], 'weib_lmb':train_df['weib_lmb'][i],
                        'lomax_k':train_df["lomax_k"][i], 'lomax_lmb':train_df['lomax_lmb'][i]},
                        ignore_index=True)
    bad_df = res_df[abs(res_df["actual_alpha"]-res_df["pred_alpha"])>3.0]
    good_df = res_df[abs(res_df["actual_alpha"]-res_df["pred_alpha"])<3.0]


def compare_loglogistic_fitting_approaches():
    """
    This experiment convinced me to abandon the Lomax
    and Weibull based LogLogistic estimation.
    """
    ti, xi = mixed_loglogistic_model()
    wbl = Weibull.est_params(ti)
    lmx = Lomax.est_params(ti)
    #Now estimate Lomax and Weibull params and construct feature vector.
    x_features = cnstrct_feature(ti)
    beta = sum(x_features*LogLogistic.lin_betas)
    alpha = sum(x_features*LogLogistic.lin_alphas)


def mixed_loglogistic_model():
    #First generate Mixed Loglogistic data.
    sampl1 = LogLogistic.samples_(1.2,300.0,5000)
    sampl2 = LogLogistic.samples_(0.7,80.0,5000)
    ti = np.concatenate((sampl1,sampl2),axis=0)
    xi = np.array([.1])
    #Time loglogistic as well
    start = time.time()
    ll = LogLogistic(ti,xi)
    end = time.time()
    print("LogLogistic gradient descent took: "+str(end-start)+ " secs")
    return ti, xi


def mixed_loglogistic_model_censored():
    sampl1 = LogLogistic.samples_(1.2,300.0,5000)
    sampl2 = LogLogistic.samples_(0.7,80.0,5000)
    m1 = np.mean(sampl1)
    m2 = np.mean(sampl2)
    ti = np.concatenate((sampl1[sampl1<m1],sampl2[sampl2<m2]),axis=0)
    xi = np.concatenate((np.ones(sum(sampl1>m1))*m1,np.ones(sum(sampl2>m2))*m2),axis=0)
    start = time.time()
    ll = LogLogistic(ti,xi)
    end = time.time()
    print("LogLogistic gradient descent took: "+str(end-start)+ " secs")
    print("The estimated parameters are:"+str(ll.alpha)+","+str(ll.beta))


def nbinom_tst():
    t_arr = np.ones(100)
    m=10; theta=.7
    ps = t_arr/(t_arr+theta)
    n_arr = nbinom.rvs(m,ps)
    nb.NBinom.loglik(n_arr,t_arr,m,theta)
    nb.NBinom.numeric_grad(n_arr,t_arr,m,theta)
    nb2=nb.NBinom2(n_arr,t_arr)
    params = nb2.gradient_descent(verbose=True)
    return sum(params-np.array([m,theta]))<1e-1

