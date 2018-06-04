import numpy as np
from misc.sigmoid import *
from distributions.basemodel import *


class LogLogistic(Base):
    def __init__(self, alp=1, beta=0.5, ti = None, xi = None, params=np.array([500,5]), gamma=0, params0=np.array([167.0, 0.3]), verbose=False):
        if ti is not None:
            self.train_org = ti
            self.train_inorg = xi
            #self.newtonRh(params = np.array([150, 5]))
            self.gradient_descent(params = params, gamma=gamma, params0=params0, verbose=verbose)
        else:
            self.train = []
            self.test = []
            self.train_org = []
            self.train_inorg = []
            self.alpha = self.lmb = alpha
            self.beta = self.k = beta
            self.params = []

    def set_params(self, alpha, beta, params=None):
        if params is not None:
            [alpha, beta] = params[:2]
        self.k = self.beta = beta
        self.lmb = self.alpha = alpha
        self.params = [alpha, beta]

    def determine_params(self, k, lmb, params):
        '''
        Determines the parameters. Defined in basemodel.py
        '''
        return super(LogLogistic, self).determine_params(k, lmb, params)

    def pdf(self,x,alpha=None,beta=None):
        [beta, alpha] = self.determine_params(beta, alpha, None)
        return (beta/alpha)*(x/alpha)**(beta-1)/(1+(x/alpha)**beta)**2

    def cdf(self,x,alpha=None,beta=None):
        [beta, alpha] = self.determine_params(beta, alpha, None)
        return 1/(1+(x/alpha)**-beta)

    def inv_cdf(self, u, alpha=None, beta=None):
        [beta, alpha] = self.determine_params(beta, alpha, None)
        return alpha*(1/u - 1)**(-1/beta)

    def samples(self, size=1000, alpha=None, beta=None):
        [beta, alpha] = self.determine_params(beta, alpha, None)
        return self.inv_cdf(np.random.uniform(size=size), alpha, beta)

    def logpdf(self,x,alpha,beta):
        [beta, alpha] = self.determine_params(beta, alpha, None)
        return np.log(beta)-np.log(alpha) +(beta-1)*(np.log(x) - np.log(alpha)) - 2*np.log(1+(x/alpha)**beta)

    def survival(self,x,alpha=None,beta=None):
        [beta, alpha] = self.determine_params(beta, alpha, None)
        return 1-self.cdf(x,alpha,beta)

    def logsurvival(self,x,alpha,beta):
        [beta, alpha] = self.determine_params(beta, alpha, None)
        return np.log(self.survival(x,alpha,beta))

    def loglik(self,t,x,alpha,beta):
        [beta, alpha] = self.determine_params(beta, alpha, None)
        return sum(self.logpdf(t,alpha,beta)) + sum(self.logsurvival(x,alpha,beta))

    def grad(self,t,x,alp=None,beta=None):
        n = len(t)
        m = len(x)
        delalp = -n*beta/alp +2*beta/alp**(beta+1) * sum(t**beta/(1+(t/alp)**beta)) + beta/alp**(beta+1)*sum(x**beta/(1+(x/alp)**beta))
        delbeta = n/beta -n*np.log(alp) + sum(np.log(t)) -2*sum((t/alp)**beta/(1+(t/alp)**beta)*np.log(t/alp) ) - sum((x/alp)**beta/(1+(x/alp)**beta)*np.log(x/alp))
        return np.array([delalp,delbeta])

    def numerical_grad(self,t,x,alp,beta):
        eps = 1e-5
        delalp = (self.loglik(t,x,alp+eps,beta) - self.loglik(t,x,alp-eps,beta))/2/eps
        delbeta = (self.loglik(t,x,alp,beta+eps) - self.loglik(t,x,alp,beta-eps))/2/eps
        return np.array([delalp,delbeta])

    '''
    def gradient_descent(self, numIter=2001, params = np.array([167.0, 0.3])):
        for i in range(numIter):
            #lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1],params[2])
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1])
            params2 = params + 1e-10*directn
            lik = self.loglik(self.train_org,self.train_inorg,params2[0],params2[1])
            for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1]:
                params1 = params + alp1 * directn
                if(min(params1) > 0):
                    lik1 = self.loglik(self.train_org,self.train_inorg,params1[0],params1[1])
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
            params = params2
            if i%100 == 0:
                print("Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn))
                print("\n########\n")
        #[self.alpha,self.beta] = params
        self.set_params(params[0], params[1], params)
        return params
    '''

    def numerical_hessian(self,t,x,k=0.5,lmb=0.3):
        eps = 1e-4
        delksq = (self.loglik(t,x,k+2*eps,lmb) + self.loglik(t,x,k-2*eps,lmb) - 2*self.loglik(t,x,k,lmb))/4/eps/eps
        dellmbsq = (self.loglik(t,x,k,lmb+2*eps) + self.loglik(t,x,k,lmb-2*eps) - 2*self.loglik(t,x,k,lmb))/4/eps/eps
        dellmbk = (self.loglik(t,x,k+eps,lmb+eps) + self.loglik(t,x,k-eps,lmb-eps) - self.loglik(t,x,k+eps,lmb-eps) - self.loglik(t,x,k-eps,lmb+eps))/4/eps/eps
        hess = np.zeros([2,2])
        hess[0,0] = delksq
        hess[1,1] = dellmbsq
        hess[0,1] = dellmbk
        hess[1,0] = dellmbk
        return hess

    def hessian(self,t,x,k=0.5,lmb=0.3):
        return self.numerical_hessian(t,x,k,lmb)

    '''
    def newtonRh(self, numIter=21, params = np.array([.1,.1])):
        steps = {1e-3:0, 0.01:0, 0.1:0, 0.5:0, 1.0:0, 2.0:0, 2.5:0, 3.0:0, 3.5:0, 3.7:0, 4.0:0, 4.5:0, 4.7:0, 5.5:0, 6.0:0, 6.5:0, 7.0:0, 7.5:0, 8.0:0, 8.5:0, 9.0:0, 9.5:0, 10.0:0, 12.0:0, 15.0:0, 20.0:0, 25.0:0, 27.0:0, 35.0:0, 37.0:0, 40.0:0, 50.0:0,100.0:0,200.0:0,500.0:0,1000.0:0}
        for i in range(numIter):
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1])
            if sum(abs(directn)) < 1e-5:
                print("\nIt took: " + str(i) + " Iterations.\n Gradients - " + str(directn))
                [self.alpha, self.beta] = params
                self.alp = self.alpha
                self.params = params
                return params
            lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1])
            step = np.linalg.solve(self.hessian(self.train_org,self.train_inorg,params[0],params[1]),directn)
            params2 = params - 1e-6 * step
            scale = 1e-3
            for alp1 in steps.keys():
                params1 = params - alp1 * step
                if max(params1) > 0:
                    lik1 = self.loglik(self.train_org,self.train_inorg,params1[0],params1[1])
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
                        scale = alp1
            steps[scale] = steps[scale] + 1
            params = params2
            if i % 10 == 0:
                print("Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn) + "\n##\n\n")
        #[self.alpha, self.beta] = params
        self.set_params(params[0], params[1], params)
        print(steps)
        return params
    '''

def fixedAlp(beta):
    alp = 79.82
    return ll.grad(ll.train_org,ll.train_inorg,alp,beta)[1]

def fixedBeta(alp):
    beta = 1.35
    return ll.grad(ll.train_org,ll.train_inorg,alp,beta)[0]

def bisection(bisection_fn, a=1e-6, b=2000):
    n=1
    while n<10000:
        c=(a+b)/2
        if bisection_fn(c)==0 or (b-a)/2<1e-6:
            return c
        n = n+1
        if (bisection_fn(c) > 0) == (bisection_fn(a) > 0):
            a=c
        else:
            b=c


