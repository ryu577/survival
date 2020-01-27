import numpy as np
import abc


class BaseMix(object):
    __metaclass__ = abc.ABCMeta

    def numr_grad_prms_(self, prms):
        """
        Calculates the numerical gradient given
        a paramter array as input.
        """
        eps = 1e-5
        grd = np.zeros(len(prms))
        for i in range(len(grd)):
            prms[i] -= eps
            lik1 = self.loglik_prms(prms)
            prms[i] += eps
            lik2 = self.loglik_prms(prms)
            grd[i] = (lik2 - lik1) / 2 / eps
            prms[i] += eps
        return grd

    @abc.abstractmethod
    def loglik_prms(self, prms):
        pass

