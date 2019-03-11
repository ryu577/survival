import numpy as np
import abc
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import exponweib


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


