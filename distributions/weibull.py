import numpy as np
from scipy.stats import exponweib
from misc.sigmoid import *
from distributions.basemodel import *

class Weibull(Base):
    '''
    Methods pertaining to the Weibull distribution.
    This is currently the only distribution that supports
    regressing features.
    '''
    def __init__(self,k=None,lmb=None,ti=None,xi=None):
        '''
        Initializes an instance of the Weibull distribution.
        '''
        if ti is not None:
            self.train_org = ti
            self.train_inorg = xi
            self.t = ti
            self.x = xi
            self.x_samples = None
            self.x_censored = None
            [self.k, self.lmb] = self.gradient_descent(params=np.array([.5,.3]))
        else:
            self.train = []
            self.test = []
            self.train_org = []
            self.train_inorg = []
            self.k = k
            self.lmb = lmb
            self.params = []
            x_samples = generate_features(100)
            t = generate_weibull(100)
            self.x_censored = x_samples[t>1.5,]
            self.x_samples = x_samples[t<1.5,]
            self.x = np.ones(sum(t>1.5))*1.5
            self.t = t[t<1.5]

    def determine_params(self, k, lmb, params):
        '''
        Sets the parameters. Inherited from basemodel
        args:
            k: The shape parameter.
            lmb: The scale parameter.
            params: An array of shape and scale parameters.
        '''
        return super(Weibull, self).determine_params(k, lmb, params)

    def logpdf(self,x,k,lmb):
        '''
        The logarithm of the PDF function.
        args:
            x: Value at which to evaluate.
            k: Shape parameter.
            lmb: Scale parameter.
        '''
        with np.errstate(all='ignore'):
            return np.log(k) - k*np.log(lmb) + (k-1)*np.log(x) - (x/lmb)**k

    def pdf(self,x,k=-1,lmb=-1,params=None):
        '''
        The probability density function of the Weibull.
        args:
            x: The value at which to evaluate.
            k: The shape parameter.
            lmb: The scale parameter.            
        '''
        [k,lmb] = self.determine_params(k,lmb,params)
        return k/lmb*(x/lmb)**(k-1)*np.exp(-(x/lmb)**k)

    def pdf_grad(self,x,k,lmb):
        '''
        The gradient of the PDF.
        args:
            x: The value at which to evaluate.
            k: The shape parameter.
            lmb: The scale parameter.
        '''
        delWeibullDelLmb = (1-(x/lmb)**k)*(-k/lmb)*self.pdf(x, k, lmb)
        delWeibullDelK = self.pdf(x, k, lmb)*( (-(x/lmb)**k+1)*np.log(x/lmb) + 1/k)
        return np.array([delWeibullDelK,delWeibullDelLmb])

    def cdf(self,t,k=-1,lmb=-1,params=None):
        '''
        The cumulative distribution function.
        args:
            t: The value at which to evaluate.
            k: The shape parameter.
            lmb: The scale parameter.
        '''
        return 1 - self.survival(t,k,lmb)

    def survival(self,t,k=-1,lmb=-1,params=None):
        '''
        The survival function of the distribution (1-cdf)
        args:
            t: The value at which to evaluate.
            k: The shape parameter.
            lmb: The scale parameter.
        '''
        [k,lmb] = self.determine_params(k,lmb,params)
        return np.exp(-(t/lmb)**k)

    def survival_grad(self,x,k,lmb):
        '''
        The gradient of the survival function.
        args:
            x: The value at which to evaluate.
            k: The shape parameter.
            lmb: The scale parameter.
        '''
        survive = self.survival(x,k,lmb)
        delk = -survive * (x/lmb)**k * np.log(x/lmb)
        dellmb = survive * (x/lmb)**k * (k/lmb)
        return np.array([delk,dellmb])

    def logsurvival(self,t,k=-1,lmb=-1,params=None):
        '''
        The logarithm of the survival function.
        args:
            t: The value at which to evaluate.
            k: The shape parameter.
            lmb: The scale parameter.
            params: An array of shape and scale parameters.
        '''
        with np.errstate(all='ignore'):
            return -(t/lmb)**k

    def hazard(self,x,k,lmb):
        '''
        The hazard rate of the Weibull (probability of seeing
        a sample in next time unit conditional on not having
        seen a sample until current time).
        args:
            x: The value at which to evaluate.
            k: The shape parameter.
            lmb: The scale parameter.
        '''
        [k,lmb] = self.determine_params(k,lmb,params)
        return self.pdf(x,k,lmb)/self.survival(x,k,lmb)

    def loglik(self,t,x = np.array([0]),k=0.5,lmb=0.3,W=None,x_samples=None,x_censored=None):
        '''
        The loglikelihood function for some observed data. Used to fit parameters to the data.
        Ability to regress against features optional.
        args:
            t: The observed samples.
            x: The censored samples.
            k: The starting value of the shape parameter.
            lmb: The starting value of the scale parameter.
            W: In case of regressing with features, this
               is a matrix of the parameters. It transforms
               the feature space to the shape and scale
               parameters of the Weibull.
            x_samples: The matrix of features associated 
                       with the sampled observations (t).
                       Should have the same rows as size of t.
            x_censored: The matrix of features associated
                        with the censored observations (x).
                        Should have the same rows as size of x.

        '''
        #
        ## If there are features, calculate gradient of features.
        #
        if W is not None and len(W.shape)==2 and x_samples is not None and x_censored is not None:
            lik = 0
            s1 = Sigmoid(6.0)
            s2 = Sigmoid(1000.0)

            for i in range(len(x_samples)):
                theta = np.dot(W.T,x_samples[i])
                [k, lmb] = [s1.transformed(theta[0]), s2.transformed(theta[1])]
                lik += self.logpdf(t[i],k,lmb)
            for i in range(len(x_censored)):
                theta = np.dot(W.T,x_censored[i])
                [k, lmb] = [s1.transformed(theta[0]), s2.transformed(theta[1])]
                lik += self.logsurvival(x[i],k,lmb)
            return lik
        #
        ## If there are no features, calculate feature-less gradients.
        #
        else:
            return sum(self.logpdf(t, k, lmb)) + sum(self.logsurvival(x, k, lmb))

    def grad(self,t,x= np.array([1e-3]),k=0.5,lmb=0.3,W=None,x_samples=None,x_censored=None):
        '''
        The gradient of the loglikelihood function for some observed data. 
        Used to fit parameters to the data.
        Ability to regress against features optional.
        args:
            t: The observed samples.
            x: The censored samples.
            k: The starting value of the shape parameter.
            lmb: The starting value of the scale parameter.
            W: In case of regressing with features, this
               is a matrix of the parameters. It transforms
               the feature space to the shape and scale
               parameters of the Weibull.
            x_samples: The matrix of features associated 
                       with the sampled observations (t).
                       Should have the same rows as size of t.
            x_censored: The matrix of features associated
                        with the censored observations (x).
                        Should have the same rows as size of x.

        '''
        #
        ## If there are features, calculate likelihood with the help of features.
        #
        if W is not None and len(W.shape)==2 and x_samples is not None and x_censored is not None:
            delW = np.zeros(W.shape)
            s1 = Sigmoid(6.0)
            s2 = Sigmoid(1000.0)

            for i in range(len(x_samples)):
                theta = np.dot(W.T,x_samples[i])
                [k, lmb] = [s1.transformed(theta[0]), s2.transformed(theta[1])]
                deltheta = np.array([s1.grad(theta[0]), s2.grad(theta[1])]) * self.pdf_grad(t[i],k,lmb)
                pdf = self.pdf(t[i], k, lmb)
                if pdf > 1e-15: # If the pdf is zero, we need to switch to survival.
                    delW += 1/pdf * np.outer(x_samples[i],deltheta)
                else: # Now, all we will say is that recovery took more than 10 seconds.
                    deltheta = np.array([s1.grad(theta[0]), s2.grad(theta[1])]) * self.survival_grad(10.0, k, lmb)
                    delW += 1/self.survival(10.0, k, lmb) * np.outer(x_samples[i], deltheta)

            for i in range(len(x_censored)):
                theta = np.dot(W.T,x_censored[i])
                [k, lmb] = [s1.transformed(theta[0]), s2.transformed(theta[1])]
                deltheta = np.array([s1.grad(theta[0]), s2.grad(theta[1])]) * self.survival_grad(x[i], k, lmb)
                sur = self.survival(x[i], k, lmb)
                if sur > 1e-15:
                    delW += 1/sur * np.outer(x_censored[i], deltheta)
                else:
                    deltheta = np.array([s1.grad(theta[0]), s2.grad(theta[1])]) * self.survival_grad(10.0, k, lmb)
                    delW += 1/self.survival(10.0, k, lmb) * np.outer(x_censored[i], deltheta)
            return delW
        #
        ## If there are no features, calculate feature-less likelihood.
        #
        else:
            n = len(t)
            delk = n/k - n*np.log(lmb) + sum(np.log(t)) - sum((t/lmb)**k*np.log(t/lmb)) - sum((x/lmb)**k*np.log(x/lmb))
            dellmb = -n*k/lmb + k/(lmb**(k+1)) * (sum(t**k) + sum(x**k))
            return np.array([delk,dellmb])

    def get_params(self,W):
        '''
        In the case of regressing against features, 
        we want to ensure that the final features
        are always positive. For this, we apply 
        sigmoid functions to the output of what we
        get from multiplying the parametres matrix
        with the feature vector.
        args:
            W: The parameter matrix.
        '''
        theta = np.dot(W.T,x[i])
        kappa = softmax(theta[0],6.0)
        lmb = softmax(theta[1],1000.0)
        return np.array([kappa,lmb])

    def numerical_grad(self,t,x,k=0.5,lmb=0.3,W=None,x_samples=None,x_censored=None):
        '''
        The numericalgradient of the loglikelihood function for some observed data. 
        Used to validate the analytic gradient.
        args:
            t: The observed samples.
            x: The censored samples.
            k: The starting value of the shape parameter.
            lmb: The starting value of the scale parameter.
            W: In case of regressing with features, this
               is a matrix of the parameters. It transforms
               the feature space to the shape and scale
               parameters of the Weibull.
            x_samples: The matrix of features associated 
                       with the sampled observations (t).
                       Should have the same rows as size of t.
            x_censored: The matrix of features associated
                        with the censored observations (x).
                        Should have the same rows as size of x.

        '''
        eps = 1e-5
        #
        ## If there are features, calculate likelihood with the help of features.
        #
        if W is not None and len(W.shape) == 2 and x_samples is not None and x_censored is not None:
            delW = np.zeros(W.shape)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W[i,j] = W[i,j] + eps
                    hi = self.loglik(t,x,k,lmb,W,x_samples,x_censored)
                    W[i,j] = W[i,j] - 2*eps
                    lo = self.loglik(t,x,k,lmb,W,x_samples,x_censored)
                    delW[i,j] = (hi-lo)/2/eps
                    W[i,j] = W[i,j] + eps
            return delW
        #
        ## If there are no features, calculate feature-less likelihood.
        #
        else:
            delk = (self.loglik(t,x,k+eps,lmb) - self.loglik(t,x,k-eps,lmb))/2/eps
            dellmb = (self.loglik(t,x,k,lmb+eps) - self.loglik(t,x,k,lmb-eps))/2/eps
            return np.array([delk,dellmb])

    '''
    def gradient_descent(self, numIter=2001, params = np.array([.5,.3])):
        for i in range(numIter):
            #lik = self.loglik(self.t, self.x, params[0], params[1], params, self.x_samples, self.x_censored)
            directn = self.grad(self.t, self.x, params[0], params[1], params, self.x_samples, self.x_censored)
            params2 = params + 1e-9*directn
            lik = self.loglik(self.t, self.x, params[0], params[1], params2, self.x_samples, self.x_censored)
            for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1]:
                params1 = params + alp1 * directn
                if len(params1.shape) == 2 or min(params1) > 0:
                    lik1 = self.loglik(self.t, self.x, params1[0], params1[1], params1, self.x_samples, self.x_censored)
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
            params = params2
            if i%25 == 0:
                print("Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn))
                print("\n########\n")
        return params
    '''

    
    def hessian(self,t,x,k=0.5,lmb=0.3):
        '''
        The hessian of the log likelihood function.
        Used by the Newton Raphson method.
        args:
            t: The array of observations of arrival times.
            x: The array of censored observations.
            k: The shape parameter.
            lmb: The scale parameter.
        '''
        n = len(t)
        delksq = -n/k**2 - sum( (t/lmb)**k*np.log(t/lmb)**2 ) - sum( (x/lmb)**k*np.log(x/lmb)**2 )
        dellmbsq = n*k/lmb**2 + ( sum(t**k) + sum(x**k) )*(- k*(k+1)/lmb**(k+2))
        dellmbk = -n/lmb + 1/lmb* ( sum( k*(t/lmb)**k*np.log(t/lmb) + (t/lmb)**k ) +  sum( k*(x/lmb)**k*np.log(x/lmb) + (x/lmb)**k ) )
        hess = np.zeros([2,2])
        hess[0,0] = delksq
        hess[1,1] = dellmbsq
        hess[0,1] = hess[1,0] = dellmbk
        return hess
    

    def lmbd(self,t,x,k):
        n = len(x)
        return ((sum(t**k) + sum(x**k)) / n)**(1/k)

    def kappa(self,t,x,k):
        n = len(t)
        return n/k + sum(np.log(t)) - n*(sum(t**k*np.log(t)) + sum(x**k*np.log(x)))/(sum(x**k) + sum(t**k))

    def bisection(self,a=1e-6,b=1000):
        n=1
        while n<10000:
            c=(a+b)/2
            if self.kappa(self.train_org,self.train_inorg,c)==0 or (b-a)/2<1e-6:
                return c
            n = n + 1
            if (self.kappa(self.train_org,self.train_inorg,c) > 0) == (self.kappa(self.train_org,self.train_inorg,c) > 0):
                a=c
            else:
                b=c

    def optimal_wait_threshold(self, intervention_cost):
        '''
        Given the cost of plan B (intervention_cost), what
        is the optimal time we should wait for this distribution?
        args:
            intervention_cost: The cost of giving up on this distribution.
        '''
        return self.lmb ** (self.k / (self.k - 1)) / (intervention_cost * self.k) ** (1 / (self.k - 1))

    def samples(self, size = 1000):
        '''
        Generates samples from current Weibull distribution.
        args:
            size: The number of samples to be generated.
        '''
        return exponweib.rvs(a=1,c=self.k,scale=self.lmb,size=size)



def generate_features(size):
    x1 = np.array([[1,1,0],[1,1,0]])
    x1 = np.repeat(x1,[2,(size-2)],axis=0)
    x2 = np.array([[1,0,1],[1,0,1]])
    x2 = np.repeat(x2,[2,(size-2)],axis=0)
    x = np.concatenate([x1,x2],axis=0)
    return x

def generate_weibull(size):
    k1 = 0.5
    scale1 = 0.3
    dat1 = exponweib.rvs(a=1,c=k1,scale=scale1,size=size)
    k2 = 1.1
    scale2 = 0.1
    dat2 = exponweib.rvs(a=1,c=k2,scale=scale2,size=size)
    dat = np.concatenate([dat1,dat2])
    return dat

if __name__ == '__main__':
    w = Weibull()
    x_samples = generate_features(100)
    t = generate_weibull(100)
    x_censored = x_samples[t>1.5,]
    x_samples = x_samples[t<1.5,]
    x = np.ones(sum(t>1.5))*1.5
    t = t[t<1.5]
    W = np.array([[0.1,0.4],[0.5,0.3],[0.2,0.7]])
    print(str(w.loglik(t,x,W=W,x_censored=x_censored,x_samples=x_samples)))
    print(str(w.grad(t,x,W=W,x_censored=x_censored,x_samples=x_samples)))
    w.gradient_descent(params=W)


