import numpy as np


class Exponential():
    def __init__(self, ts, xs=None):
        if xs is not None:
            numrtr = sum(ts)+sum(xs)
            self.lmb = numrtr/len(ts)

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
        integ_dt = (np.exp(-mu*tau1)-np.exp(-mu*tau2)\
            -np.exp(-mu*tau1)*(tau2-tau1))/mu
        integ_tdt = (-tau2*np.exp(-mu*tau2)+tau1*np.exp(-mu*tau1))/mu -\
                (np.exp(-mu*tau2)-np.exp(-mu*tau1))/mu**2
        integ_dt = -1*integ_dt
        integ_tdt = -1*integ_tdt
        a = (2*integ_tdt-(tau1+tau2)*integ_dt)/(0.6667*(tau2**3-tau1**3)-\
                0.5*(tau2**2-tau1**2)*(tau2+tau1))
        b = integ_dt/(tau2-tau1) - a*(tau2+tau1)/2
        return a, b

