import abc
import numpy as np
from survival.ptprocesses.base import Base
from scipy.stats import nbinom


class NBinom(Base):
    __metaclass__ = abc.ABCMeta
    """
    A negative binomial point process, obtained
    by mixing a Poisson process with a Gamma.
    """
    def __init__(self, m, theta):
        self.m = m
        self.theta = theta
        self.n_t = NBinom_Nt(m, theta)
        self.i_t = NBinom_T(m, theta)

    @staticmethod
    def loglik(n_arr, t_arr, m, theta):
        if len(n_arr) != len(t_arr):
            raise ValueError("Length of arrays should be the same.")
        ll = 0
        for i in range(len(n_arr)):
            p = t_arr[i]/(t_arr[i]+theta)
            ll += np.log(nbinom.pmf(n_arr[i], m, p))
        return ll

    @staticmethod
    def numeric_grad(n_arr, t_arr, m, theta, eps=1e-5):
        ll_pl = NBinom.loglik(n_arr, t_arr, m+eps, theta)
        ll_mi = NBinom.loglik(n_arr, t_arr, m-eps, theta)
        delm = (ll_pl-ll_mi)/2/eps
        ll_pl = NBinom.loglik(n_arr, t_arr, m, theta+eps)
        ll_mi = NBinom.loglik(n_arr, t_arr, m, theta-eps)
        deltheta = (ll_pl-ll_mi)/2/eps
        return np.array([delm, deltheta])


class NBinom2(NBinom):
    def __init__(self, n_arr, t_arr):
        self.n_arr = n_arr
        self.t_arr = t_arr

    def loglik(self, params):
        return NBinom.loglik(self.n_arr, self.t_arr,
                             params[0], params[1])

    def grad(self, params):
        return \
            NBinom.numeric_grad(self.n_arr, self.t_arr,
                                params[0], params[1])

class NBinom_Nt(NBinom):
    def __init__(self, m, theta):
        # return super(NBinom_Nt,self).__init__(m,theta)
        self.m = m
        self.theta = theta

    def pmf(self, k, t):
        p = t/(t+self.theta)
        return nbinom.pmf(k, self.m, p)

    def logpmf(self, k, t):
        p = t/(t+self.theta)
        return nbinom.logpmf(k, self.m, p)


class NBinom_T(NBinom):
    def __init__(self, m, theta):
        # return super(NBinom_Nt,self).__init__(m,theta)
        self.m = m
        self.theta = theta


def sample_usage():
    t_arr = np.ones(100)
    m = 10 
    theta = .7
    ps = t_arr/(t_arr+theta)
    n_arr = nbinom.rvs(m, ps)
    NBinom.loglik(n_arr, t_arr, m, theta)
    NBinom.numeric_grad(n_arr, t_arr, m, theta)
    nb2 = NBinom2(n_arr, t_arr)
    params = nb2.gradient_descent(verbose=True)
    return sum(params-np.array([m, theta])) < 1e-1
