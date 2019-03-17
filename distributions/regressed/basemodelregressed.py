import numpy as np
import abc
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import exponweib
from misc.sigmoid import *
from distributions.basemodel import *

class BaseRegressed(object):
    __metaclass__ = abc.ABCMeta

    def set_state(self, t, x, fsamples, fcensored, basedistr):
        self.t = t
        self.x = x
        self.w = None
        ## TODO: Add checks on the sizes of the two feature arrays.
        self.fsamples = fsamples
        self.fcensored = fcensored
        self.basedistr = basedistr

    @staticmethod
    def generate_features_(size1, feature):
        x1 = np.array([feature, feature])
        x1 = np.repeat(x1, [2, (size1 - 2)], axis=0)
        return x1

    @staticmethod
    def generate_data_(distribn, size, k1=1.2, scale1=300, k2=0.7, scale2=80.0):
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

    def generate_data(self, size, k1=1.2, scale1=300, k2=0.7, scale2=80.0):
        self.t, self.x, self.fsamples, self.fcensored \
            = BaseRegressed.generate_data_(type(self.basedistr), size, k1, scale1, k2, scale2)

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
            shapefn = lambda x: Sigmoid.transform_(x,5.0)
        if scalefn is None:
            scalefn = lambda x: Sigmoid.transform_(x,900.0)
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

    def loglik(self, w):
        return BaseRegressed.loglikelihood_(self.t, self.x, self.fsamples, self.fcensored,
                                self.basedistr, w)

    @staticmethod
    def numerical_grad_(ti, xi, fsamples, fcensored, distr, w, 
            shapefn=None, scalefn=None):
        numerical_grd = np.zeros(w.shape)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i,j] += 1e-5
                ll1 = BaseRegressed.loglikelihood_(ti, xi, fsamples, fcensored, 
                                                distr, w,
                                                shapefn, scalefn)
                w[i,j] -= 2e-5
                ll2 = BaseRegressed.loglikelihood_(ti, xi, fsamples, fcensored, 
                                                distr, w,
                                                shapefn, scalefn)
                w[i,j] += 1e-5
                numerical_grd[i,j] = (ll1-ll2)/2e-5
        return numerical_grd

    @staticmethod
    def grad_(ti, xi, fsamples, fcensored, distr, w, 
            shapefn=None, scalefn=None,
            shapefnder=None, scalefnder=None):
        if shapefnder is None:
            shapefnder = lambda x : Sigmoid.grad_(x,5.0)
        if scalefnder is None:
            scalefnder = lambda x : Sigmoid.grad_(x,900.0)
        if shapefn is None:
            shapefn = lambda x: Sigmoid.transform_(x,5.0)
        if scalefn is None:
            scalefn = lambda x: Sigmoid.transform_(x,900.0)
        gradw = np.zeros(w.shape)
        for i in range(len(ti)):
            currrow = fsamples[i]
            theta = np.dot(w,currrow)
            shape, scale = shapefn(theta[0]), scalefn(theta[1])
            lpdfgrd = distr.grad_l_pdf_(ti[i], shape, scale)
            fn_grad = np.array([shapefnder(theta[0]), scalefnder(theta[1])])
            deltheta = lpdfgrd*fn_grad
            gradw += np.outer(deltheta, currrow)
        for i in range(len(xi)):
            currrow = fcensored[i]
            theta = np.dot(w,currrow)
            shape, scale = shapefn(theta[0]), scalefn(theta[1])
            lpdfgrd = distr.grad_l_survival_(xi[i], shape, scale)
            fn_grad = np.array([shapefnder(theta[0]), scalefnder(theta[1])])
            deltheta = lpdfgrd*fn_grad
            gradw += np.outer(deltheta, currrow)
        return gradw

    def grad(self, w):
        return BaseRegressed.grad_(self.t, self.x, self.fsamples, self.fcensored,
                                self.basedistr, w)

    def gradient_descent(self, params, numIter=2001, verbose=False,
        step_lengths=[1e-8, 1e-7, 1e-5, 1e-3, 1e-2, .1, 10, 50, 70, 120, 150, 
                      200, 250, 270, 300, 500, 1e3, 1.5e3, 2e3, 3e3]
        ):
        '''
        Performs gradient descent to fit the parameters of our distribution.
        TODO: Make base distributions conform to this method. Then, unify all gradient descent.
        args:
            numIter: The number of iterations gradient descent should run for.
            params: The starting parameters where it starts.
            verbose: To print progress in iterations or not.            
            step_lengths: The step lengths along the gradient the algorithm should check 
                          and make the step with the best improvement in objective function.
        '''
        self.step_lens = {}
        for i in range(numIter):
            #lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1],params[2])
            directn = self.grad(params)
            if np.max(abs(directn)) < 1e-3:
                self.set_params(params[0], params[1], params)
                self.final_loglik = self.loglik(params)
                return params
            # In 20% of the iterations, we set all but one of the gradient
            # dimensions to zero.
            # This works better in practice.
            if i % 100 > 60:
                # Randomly set one coordinate to zero.
                directn[np.random.choice(len(params), 1)[0]] = 0
            params2 = params + 1e-10 * directn
            lik = self.loglik(params)
            alp_used = step_lengths[0]
            for alp1 in step_lengths:
                params1 = params + alp1 * directn
                lik1 = self.loglik(params1)
                if(lik1 > lik and np.isfinite(lik1)):
                    lik = lik1
                    params2 = params1
                    alp_used = alp1
            if alp_used in self.step_lens:
                self.step_lens[alp_used] += 1
            else:
                self.step_lens[alp_used] = 1
            params = params2
            if i % 100 == 0 and verbose:                
                print("Itrn " + str(i) + " ,obj fn: " + str(lik) + " \nparams = " + 
                    str(params) + " \ngradient = " + str(directn) + 
                        "\nstep_len=" + str(alp_used))
                print("\n########\n")
        self.set_params(params[0], params[1], params)
        self.final_loglik = lik
        return params

    def set_params(self, shape, scale, params):
        ##Ignore the first two arguments.
        self.w = params

