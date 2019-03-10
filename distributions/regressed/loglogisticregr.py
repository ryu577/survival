import numpy as np
from misc.sigmoid import *
from distributions.loglogistic import *

class LogLogisticRegr():
    def __init__(self, ll):
        self.ll = ll
        self.shapeupper=5.0
        self.scaleupper=100.0

    @staticmethod
    def loglikelihood(ti, xi, fsamples, fcensored, w, 
            shapefn=None, scalefn=None):
        """
        Calculates the loglikelihood of the model with features.
        args:
            ti: The vector of organic recoveries.
            xi: The vector of inroganic recoveries.
            fsamples: The matrix of features corresponding to organic recoveries.
            fcensored: The matrix of features corresponding to inorganic.
        """
        lik = 0
        if shapefn is None:
            shapefn = lambda x: Sigmoid.transform(x,5.0)
        if scalefn is None:
            scalefn = lambda x: Sigmoid.transform(x,900.0)
        for i in range(len(ti)):
            currentrow = fsamples[i]
            theta = np.dot(w,currentrow)
            shape = shapefn(theta[0])
            scale = scalefn(theta[1])
            lik += LogLogistic.logpdf_(ti[i], shape, scale)
        for i in range(len(xi)):
            currentrow = fcensored[i]
            theta = np.dot(w,currentrow)
            shape = shapefn(theta[0])
            scale = scalefn(theta[1])
            lik += LogLogistic.logsurvival_(xi[i], shape, scale)
        return lik

    @staticmethod
    def grad(ti, xi, fsamples, fcensored, w, shapefnder, scalefnder):
        gradw = np.zeros(w.shape)
        for i in range(len(ti)):
            currrow = fsamples[i]
            theta = np.dot(w,currrow)
            lpdfgrd = LogLogistic.grad_l_pdf_(t[i], theta[0], theta[1])
            fn_grad = np.array([shapefnder(theta[0]), scalefnder(theta[1])])
            deltheta = lpdfgrd*fn_grad
            gradw += np.outer(deltheta, currrow)
        for i in range(len(xi)):
            currrow = fcensored[i]
            theta = np.dot(w,currrow)
            lpdfgrd = LogLogistic.grad_l_survival_(x[i], theta[0], theta[1])
            fn_grad = np.array([shapefnder(theta[0]), scalefnder(theta[1])])
            deltheta = lpdfgrd*fn_grad
            gradw += np.outer(deltheta, currrow)
        return gradw


