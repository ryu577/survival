import numpy as np
from scipy.stats import lomax
from distributions.basemodel import *
from optimization.optimizn import bisection

class Lomax(Base):
    '''
    We can instantiate a Lomax distribution 
    (https://en.wikipedia.org/wiki/Lomax_distribution)
    with this class.
    '''

    def __init__(self, k=None, lmb=None, ti=None, xi=None):
        '''
        Instantiate a Lomax distribution.
        args:
            k: The shape parameter of the Lomax distribution.
            lmb: The scale parameter of the lomax distribution.
            ti: The uncensored samples for fitting the distribution.
            xi: The censored samples for fitting the distribution.
        '''
        if ti is not None:
            self.train_org = ti
            self.train_inorg = xi
            # Initialize k and lmb to some reasonable guess
            self.k = 0.85
            self.lmb = 1e-3
            self.newtonRh()
        else:
            self.train = []
            self.test = []
            self.train_org = []
            self.train_inorg = []
            self.k = k
            self.lmb = lmb
        self.params = [self.k, self.lmb]

    def determine_params(self, k, lmb, params):
        '''
        Determines the parameters. Defined in basemodel.py
        '''
        return super(Lomax, self).determine_params(k, lmb, params)

    def pdf(self, t, k=None, lmb=None, params=None):
        '''
        The probability distribution function (PDF) of the Lomax distribution.
        args:
            t: The value at which the PDF is to be calculated.
            k: The shape parameter of the Lomax distribution.
            lmb: The scale parameter of the lomax distribution.
        '''
        [k, lmb] = self.determine_params(k, lmb, params)
        return lmb * k / (1 + lmb * t)**(k + 1)

    def cdf(self, t, k=None, lmb=None, params=None):
        '''
        The cumulative density functino of the Lomax distribution.
        Probability that the distribution is lower than a certain value.
        args:
            t: The value at which CDF is to be calculated.
            k: The shape parameter of the Lomax.
            lmb: The sclae parameter of the Lomax.
            params: A 2d array with the shape and scale parameters.
        '''
        [k, lmb] = self.determine_params(k, lmb, params)
        return 1 - (1 + lmb * t)**-k

    def survival(self, t, k=None, lmb=None, params=None):
        '''
        The survival function for the Lomax distribution.
        '''
        [k, lmb] = self.determine_params(k, lmb, params)
        return (1 + lmb * t)**-k

    def logpdf(self, t, k=None, lmb=None, params=None):
        '''
        The logarithm of the PDF function. Handy for calculating log likelihood.
        args:
            t: The value at which function is to be calculated.
            l: The shape parameter.
            lmb: The scale parameter.
        '''
        [k, lmb] = self.determine_params(k, lmb, params)
        return np.log(k) + np.log(lmb) - (k + 1) * np.log(1 + lmb * t)

    def logsurvival(self, t, k=None, lmb=None, params=None):
        '''
        The logarithm of the survival function. Handy for calculating log likelihood.
        args:
            t: The value at which function is to be calculated.
            l: The shape parameter.
            lmb: The scale parameter.
        '''
        [k, lmb] = self.determine_params(k, lmb, params)
        return -k * np.log(1 + lmb * t)

    def loglik(self, t, x, k=None, lmb=None, params=None):
        '''
        The logarithm of the likelihood function.
        args:
            t: The un-censored samples.
            x: The censored samples.
            l: The shape parameter.
            lmb: The scale parameter.
        '''
        [k, lmb] = self.determine_params(k, lmb, params)
        return sum(self.logpdf(t, k, lmb)) + sum(self.logsurvival(x, k, lmb))

    def grad(self, t, x, k=0.5, lmb=0.3):
        '''
        The gradient of the log-likelihood function.
        args:
            t: The un-censored samples.
            x: The censored samples.
            l: The shape parameter.
            lmb: The scale parameter.
        '''
        n = len(t)
        m = len(x)
        delk = n / k - sum(np.log(1 + lmb * t)) - sum(np.log(1 + lmb * x))
        dellmb = n / lmb - (k + 1) * sum(t / (1 + lmb * t)
                                         ) - k * sum(x / (1 + lmb * x))
        return np.array([delk, dellmb])

    def numerical_grad(self, t, x, k=None, lmb=None):
        '''
        Calculates the gradient of the log-likelihood function numerically.
        args:
            t: The survival data.
            x: The censored data.
            k: The shape parameter.
            lmb: The scale parameter.
        '''
        if k is None or lmb is None:
            k = self.k
            lmb = self.lmb
        eps = 1e-5
        delk = (self.loglik(t, x, k + eps, lmb) -
                self.loglik(t, x, k - eps, lmb)) / 2 / eps
        dellmb = (self.loglik(t, x, k, lmb + eps) -
                  self.loglik(t, x, k, lmb - eps)) / 2 / eps
        return np.array([delk, dellmb])

    def hessian(self, t, x, k=0.5, lmb=0.3):
        '''
        The hessian of the Loglikelihood function for Lomax.
        args:
            t: The un-censored samples.
            x: The censored samples.
            l: The shape parameter.
            lmb: The scale parameter.
        '''
        n = len(t)
        delksq = -n / k**2
        dellmbsq = -n / lmb**2 + \
            (k + 1) * sum((t / (1 + lmb * t))**2) + \
            k * sum((x / (1 + lmb * x))**2)
        delklmb = -sum(t / (1 + lmb * t)) - sum(x / (1 + lmb * x))
        hess = np.zeros([2, 2])
        hess[0, 0] = delksq
        hess[1, 1] = dellmbsq
        hess[0, 1] = hess[1, 0] = delklmb
        return hess

    def numerical_hessian(self, t, x, k=0.5, lmb=0.3):
        '''
        Calculates the hessian of the log-likelihood function numerically.
        args:
            t: The survival data.
            x: The censored data.
            k: The shape parameter.
            lmb: The scale parameter.
        '''
        eps = 1e-4
        delksq = (self.loglik(t, x, k + 2 * eps, lmb) + self.loglik(t, x,
                                                                    k - 2 * eps, lmb) - 2 * self.loglik(t, x, k, lmb)) / 4 / eps / eps
        dellmbsq = (self.loglik(t, x, k, lmb + 2 * eps) + self.loglik(t, x,
                                                                      k, lmb - 2 * eps) - 2 * self.loglik(t, x, k, lmb)) / 4 / eps / eps
        dellmbk = (self.loglik(t, x, k + eps, lmb + eps) + self.loglik(t, x, k - eps, lmb - eps)
                   - self.loglik(t, x, k + eps, lmb - eps) - self.loglik(t, x, k - eps, lmb + eps)) / 4 / eps / eps
        hess = np.zeros([2, 2])
        hess[0, 0] = delksq
        hess[1, 1] = dellmbsq
        hess[0, 1] = hess[1, 0] = dellmbk
        return hess

    def gradient_descent(self, numIter=2001, params=np.array([.5, .3]), verbose=False):
        '''
        Performs gradient descent to get the best fitting parameters for
        this Lomax given the censored and un-censored data.
        args:
            numIter: The maximum number of iterations for the iterative method.
            params: The initial guess for the shape and scale parameters respectively.
            verbose: Set to true for debugging. Shows progress as it fits data.
        '''
        for i in range(numIter):
            lik = self.loglik(self.train_org, self.train_inorg,
                              params[0], params[1])
            directn = self.grad(
                self.train_org, self.train_inorg, params[0], params[1])
            params2 = params
            for alp1 in [1e-8, 1e-7, 1e-5, 1e-3, 1e-2, .1]:
                params1 = params + alp1 * directn
                if(min(params1) > 0):
                    lik1 = self.loglik(
                        self.train_org, self.train_inorg, params1[0], params1[1])
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
            params = params2
            if i % 100 == 0 and verbose:
                print("Iteration " + str(i) + " ,objective function: " + str(lik) +
                      " \nparams = " + str(params) + " \nGradient = " + str(directn))
                print("\n########\n")
        return params

    '''
    def newtonRh(self, numIter=101, params = np.array([.1,.1]), verbose=False):
        """
        Fits the parameters of a Lomax distribution to data (censored and uncensored).
        Uses the Newton Raphson method for explanation, see: https://www.youtube.com/watch?v=acsSIyDugP0
        args:
            numIter: The maximum number of iterations for the iterative method.
            params: The initial guess for the shape and scale parameters respectively.
            verbose: Set to true for debugging. Shows progress as it fits data.
        """
        for i in range(numIter):
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1])
            if sum(abs(directn)) < 1e-5:
                if verbose:
                    print("\nIt took: " + str(i) + " Iterations.\n Gradients - " + str(directn))
                self.params = params
                [self.k, self.lmb] = params
                return params
            lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1])
            step = np.linalg.solve(self.hessian(self.train_org,self.train_inorg,params[0],params[1]),directn)
            params = params - step
            if min(params) < 0:
                print("Drastic measures")
                params = params + step # undo the effect of taking the step.
                params2 = params
                for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1,.5,1.0]:
                    params1 = params - alp1 * step
                    if(max(params1) > 0):
                        lik1 = self.loglik(self.train_org,self.train_inorg,params1[0],params1[1])
                        if(lik1 > lik and np.isfinite(lik1)):
                            lik = lik1
                            params2 = params1
                            scale = alp1
                params = params2
            if i % 10 == 0 and verbose:
                print("Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn) + "\n##\n\n")
        [self.k, self.lmb] = params
        self.params = params
        return params
    '''

    def optimal_wait_threshold(self, intervention_cost, k=None, lmb=None):
        '''
        Gets the optimal time one should wait for a Lomax recovery before intervention.
        args:
            intervention_cost: The cost of intervening.
            k: The shape parameter of this Lomax distribution.
            lmb: The scale parameter of this Lomax distribution.
        '''
        if k is None or lmb is None:
            k = self.k
            lmb = self.lmb
        return (intervention_cost * k - 1 / lmb)

    def expectedDT(self, tau, k, lmb, intervention_cost):
        '''
        The expected downtime incurred when the waiting threshold is set to an arbitrary value.
        args:
            tau: The value we should set for the intervention threshold.
            k: The shape parameter of the current Lomax.
            lmb: The scale parameter of the current Lomax.
            intervention_cost: The cost of intervening.
        '''
        return 1 / lmb / (k - 1) - (1 / lmb / (k - 1) \
            + tau * k / (k - 1)) * 1 / (1 + lmb * tau)**k \
            + (tau + intervention_cost) * 1 / (1 + lmb * tau)**k

    @staticmethod
    def expectedDT_s(tau, k, lmb, intervention_cost):
        '''
        The expected downtime incurred when the waiting threshold is 
        set to an arbitrary value (static version).
        args:
            tau: The value we should set for the intervention threshold.
            k: The shape parameter of the current Lomax.
            lmb: The scale parameter of the current Lomax.
            intervention_cost: The cost of intervening.
        '''
        return 1 / lmb / (k - 1) - (1 / lmb / (k - 1) \
            + tau * k / (k - 1)) * 1 / (1 + lmb * tau)**k \
            + (tau + intervention_cost) * 1 / (1 + lmb * tau)**k

    def expectedT(self, tau, k=None, lmb=None, params=None):
        '''
        The expected value of the Lomax conditional on it being less than tau.
        args:
            tau: Censor the Lomax here.
            k: The shape parameter of the current Lomax.
            lmb: The scale parameter of the current Lomax.
            params: A 2-d array with shape and scale parameters.
        '''
        [k, lmb] = self.determine_params(k, lmb, params)
        return (1 / lmb / (k - 1) - (1 / lmb / (k - 1) \
            + tau * k / (k - 1)) * 1 / (1 + lmb * tau)**k) / (1 - 1 / (1 + lmb * tau)**k)

    def samples(self, k=None, lmb=None, size=1000, params=None):
        '''
        Generates samples for the Lomax distribution.
        args:
            k: Shape of Lomax.
            lmb: Scale of Lomax.
            size: The number of simulations to be generated.
            params: A 2-d array with shape and scale parameters.
        '''
        [k, lmb] = self.determine_params(k, lmb, params)
        return lomax.rvs(c=k, scale=(1 / lmb), size=size)

    @staticmethod
    def samples_(k, lmb, size=1000):
        return lomax.rvs(c=k, scale=(1 / lmb), size=size)

    @staticmethod
    def kappafn_k(t, x, wt=None, wx=None, lmb=0.1):
        """
        See [1]
        """
        if wt is None:
            wt = np.ones(len(t))
            wx = np.ones(len(x))
        n = sum(wt)
        return n / (sum(wt*np.log(1 + lmb * t)) + sum(wx*np.log(1 + lmb * x)))

    @staticmethod
    def kappafn_lmb(t, x, wt=None, wx=None, lmb=0.1):
        """
        See [1]
        """
        if wt is None:
            wt = np.ones(len(t))
            wx = np.ones(len(x))
        n = sum(wt)
        return (n / lmb - sum(t*wt / (1 + lmb * t))) /\
             (sum(t*wt / (1 + lmb * t)) + sum(x*wx / (1 + lmb * x)))

    @staticmethod
    def bisection_fn(lmb, t, x=np.array([]), wt=None, wx=None):
        return Lomax.kappafn_k(t, x, wt, wx, lmb) \
            - Lomax.kappafn_lmb(t, x, wt, wx, lmb)

    @staticmethod
    def est_params(t, x=np.array([]), wt=None, wx=None):
        fn = lambda lmb: Lomax.bisection_fn(lmb, t, x, wt, wx)
        lmb = bisection(fn, 0.1, 100)
        k = Lomax.kappafn_lmb(t, x, lmb)
        return k, lmb

#[1] https://onedrive.live.com/view.aspx?resid=7CAD132A61933826%216310&id=documents&wd=target%28Math%2FSurvival.one%7C33EFE553-AA82-43B5-AD47-9900633D2A1E%2FLomax%20Wts%7C0902ADB4-A003-4CFF-98E4-95870B6E7759%2F%29onenote:https://d.docs.live.net/7cad132a61933826/Documents/Topics/Math/Survival.one#Lomax%20Wts&section-id={33EFE553-AA82-43B5-AD47-9900633D2A1E}&page-id={0902ADB4-A003-4CFF-98E4-95870B6E7759}&end
