import numpy as np
import matplotlib.pyplot as plt
from survival.optimization.optimizn import *
from scipy.stats import expon
from scipy.optimize import minimize


class Exponential():
    def __init__(self, ts, xs=None):
        if xs is not None:
            denominator = sum(ts)+sum(xs)
            self.lmb = len(ts)/denominator
        else:
            self.lmb = len(ts)/sum(ts)

    @staticmethod
    def samples_(lmb, size=1000):
        return np.random.exponential(1/lmb,size=size)

    @staticmethod
    def linear_coefs_cdf(mu, tau1, tau2):
        integ_dt = (np.exp(-mu*tau1)-np.exp(-mu*tau2)\
            -np.exp(-mu*tau1)*(tau2-tau1))/mu
        integ_tdt = (-tau2*np.exp(-mu*tau2)+tau1*np.exp(-mu*tau1))/mu -\
                (np.exp(-mu*tau2)-np.exp(-mu*tau1))/mu**2
        a = (2*integ_tdt-(tau1+tau2)*integ_dt)/(0.6667*(tau2**3-tau1**3)-\
                0.5*(tau2**2-tau1**2)*(tau2+tau1))
        b = integ_dt/(tau2-tau1) - a*(tau2+tau1)/2
        return a, b

    @staticmethod
    def linear_coefs_survival(mu, tau1, tau2):
        integ_dt = (np.exp(-mu*tau1)-np.exp(-mu*tau2))/mu
        integ_tdt = (-tau2*np.exp(-mu*tau2)+tau1*np.exp(-mu*tau1))/mu -\
                (np.exp(-mu*tau2)-np.exp(-mu*tau1))/mu**2
        a = (2*integ_tdt-(tau1+tau2)*integ_dt)/(0.6667*(tau2**3-tau1**3)-\
                0.5*(tau2**2-tau1**2)*(tau2+tau1))
        b = integ_dt/(tau2-tau1) - a*(tau2+tau1)/2
        return a, b

    @staticmethod
    def mle_uncensored(t):
        return len(t)/sum(t)

    @staticmethod
    def mle_censored_full_info_loss(t, tau):
        """
        Exponential distribution where we censor the data at some 
        value, tau and don't record anything about the censored data.
        """
        n = len(t)
        fn = lambda lmb: 1/lmb - tau/(np.exp(lmb*tau)-1) - sum(t)/n
        ## The assumption is that the rate is between 1e-3 and 1e4.
        lmb = bisection(fn,1e-3,1e4)
        return lmb
    
    @staticmethod
    def fit_censored_data(x, censor):
        init_scale = 1/np.mean(x)
        def log_censored_likelihood(scale):
            return -np.sum(np.log(expon.pdf(x, loc=0, scale=scale) \
                    / expon.cdf(censor, loc=0, scale=scale)))
        scale_result = minimize(log_censored_likelihood, init_scale, method='Nelder-Mead')
        return scale_result.x[0] # convert scale to lambda


def tst_plot_survival_approx():
    mu1 = 0.012805
    mu2 = 0.008958
    xs = np.arange(0,600,1.0)
    ys1 = np.exp(-mu1*xs)
    ys2 = np.exp(-mu2*xs)
    coefs1 = Exponential.linear_coefs_survival(mu1, 10, 500)
    coefs2 = Exponential.linear_coefs_survival(mu2, 10, 500)
    ys1_lin = coefs1[0]*xs+coefs1[1]
    ys2_lin = coefs2[0]*xs+coefs2[1]
    plt.plot(xs, ys1, label='high_mu')
    plt.plot(xs, ys1_lin, label='lin_highmu')
    plt.plot(xs, ys2,label='low_mu')
    plt.plot(xs, ys2_lin,label='lin_lowmu')
    plt.legend()
    plt.show()

