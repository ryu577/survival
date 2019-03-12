import numpy as np
import abc
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import exponweib
from misc.sigmoid import *

class BaseRegressed(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def generate_features_(size1, feature):
        x1 = np.array([feature, feature])
        x1 = np.repeat(x1, [2, (size1 - 2)], axis=0)
        return x1

    @staticmethod
    def generate_data_(distribn, size, k1=1.2, scale1=300, k2=0.7, scale2=80.):
        dat1 = distribn.samples_(k1, scale1, size=size)
        censorlvl = np.mean(dat1)
        ti1 = dat1[dat1<censorlvl]
        xi1 = np.ones(sum(dat1>censorlvl))*censorlvl
        fsamples1 = BaseRegressed.generate_features_(len(ti1),[1,0,1])
        fcensored1 = BaseRegressed.generate_features_(len(xi1),[1,0,1])
        dat2 = distribn.samples_(k2, scale2, size=size)
        censorlvl = np.mean(dat2)
        ti2 = dat2[dat2<censorlvl]
        xi2 = np.ones(sum(dat2>censorlvl))*censorlvl
        fsamples2 = BaseRegressed.generate_features_(len(ti2),[1,1,0])
        fcensored2 = BaseRegressed.generate_features_(len(xi2),[1,1,0])
        ti = np.concatenate((ti1,ti2),axis=0)
        xi = np.concatenate((xi1,xi2),axis=0)
        fsamples = np.concatenate((fsamples1,fsamples2),axis=0)
        fcensored = np.concatenate((fcensored1,fcensored2),axis=0)
        return ti, xi, fsamples, fcensored

    @staticmethod
    def loglikelihood_(ti, xi, fsamples, fcensored, distr, w, 
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
            lik += distr.logpdf_(ti[i], shape, scale)
        for i in range(len(xi)):
            currentrow = fcensored[i]
            theta = np.dot(w,currentrow)
            shape = shapefn(theta[0])
            scale = scalefn(theta[1])
            lik += distr.logsurvival_(xi[i], shape, scale)
        return lik

    @staticmethod
    def numerical_grad_(ti, xi, fsamples, fcensored, distr, w, 
            shapefn=None, scalefn=None):
        numerical_grd = np.zeros(w.shape)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i,j] += 1e-4
                ll1 = BaseRegressed.loglikelihood_(ti, xi, fsamples, fcensored, distr, w,
                                                shapefn, scalefn)
                w[i,j] -= 2e-4
                ll2 = BaseRegressed.loglikelihood_(ti, xi, fsamples, fcensored, distr, w,
                                                shapefn, scalefn)
                w[i,j] += 1e-4
                numerical_grd[i,j] = (ll2-ll1)/2e-4
        return numerical_grd

    @staticmethod
    def grad_(ti, xi, fsamples, fcensored, w, distr, shapefnder, scalefnder):
        gradw = np.zeros(w.shape)
        for i in range(len(ti)):
            currrow = fsamples[i]
            theta = np.dot(w,currrow)
            lpdfgrd = distr.grad_l_pdf_(ti[i], theta[0], theta[1])
            fn_grad = np.array([shapefnder(theta[0]), scalefnder(theta[1])])
            deltheta = lpdfgrd*fn_grad
            gradw += np.outer(deltheta, currrow)
        for i in range(len(xi)):
            currrow = fcensored[i]
            theta = np.dot(w,currrow)
            lpdfgrd = distr.grad_l_survival_(xi[i], theta[0], theta[1])
            fn_grad = np.array([shapefnder(theta[0]), scalefnder(theta[1])])
            deltheta = lpdfgrd*fn_grad
            gradw += np.outer(deltheta, currrow)
        return gradw

