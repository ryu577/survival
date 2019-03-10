import numpy as np
from misc.sigmoid import *
from distributions.loglogistic import *

class LogLogisticRegr():
    def __init__(self, ll):
        self.ll = ll
        self.shapeupper=5.0
        self.scaleupper=100.0

    @staticmethod
    def loglikelihood(ti, xi, fsamples, fcensored, w, ll):
        """
        Calculates the loglikelihood of the model with features.
        args:
            ti: The vector of organic recoveries.
            xi: The vector of inroganic recoveries.
            fsamples: The matrix of features corresponding to organic recoveries.
            fcensored: The matrix of features corresponding to inorganic.
        """
        lik = 0
        for i in range(len(ti)):
            currentrow = fsamples[i]
            theta = np.dot(w,currentrow)
            shape = Sigmoid.transform(theta[0], 5.0)
            scale = Sigmoid.transform(theta[1], 100.0)
            lik += LogLogistic.logpdf_(ti[i], shape, scale)
        for i in range(len(xi)):
            currentrow = fcensored[i]
            theta = np.dot(w,currentrow)
            shape = Sigmoid.transform(theta[0], 5.0)
            scale = Sigmoid.transform(theta[1], 100.0)
            lik += LogLogistic.logsurvival_(xi[i], shape, scale)
        return lik

