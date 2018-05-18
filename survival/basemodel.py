import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


class Base(object):
    '''
    Numerically integrates the PDF and obtains the expected value of x conditional on x less than y.
    '''
    def Ex_x_le_y(self,xs = np.arange(1,100000)*0.01):
        vals = []
        # \int_0^t x f_x dx
        for i in xs:
            vals.append((i+0.005)*self.pdf(i+0.005)*0.01)
        return np.cumsum(vals)

    def expectedXBwLts(self, t1, t2, k=None, lmb=None):
        ress = integrate.quad(lambda x: x * self.pdf(x,k,lmb), t1, t2)
        prob = self.cdf(t2,k,lmb) - (self.cdf(t1,k,lmb) if t1 > 0 else 0)
        return ress[0]/prob

    '''
    Combines the expected downtime when the recovery happens before and after the wait threshold.
    '''
    def expected_downtime(self,Y,xs=np.arange(1,100000)*0.01,lmb=0,reg='log'):
        highterms = self.survival(xs)*(xs+Y)
        lowterms = self.Ex_x_le_y(xs)
        et = lowterms + highterms
        if reg == 'log':
            et += lmb*np.log(xs)
        elif reg == 'sqrt':
            et += lmb*xs**.5
        elif reg == 'sqr':
            et += lmb*xs**2
        return et

    def determine_params(self, k=-1, lmb=-1, params=None):
        if params is not None:
            k = params[0]
            lmb = params[1]
        else:
            if k is None or k < 0:
                k = self.k
            if lmb is None or lmb < 0:
                lmb = self.lmb
        return [k,lmb]

    def prob_TgrTau(self,xs=np.arange(1,100000)*0.01,lmb=0.2,t0=900.0,Y=480.0):
        return lmb*((xs>t0)*(self.survival(t0)-self.survival(xs)) + (xs> (t0-Y))*self.survival(xs))

    def expectedT(self,tau,k=None,lmb=None,params = None):
        [k,lmb] = self.determine_params(k,lmb,params)
        return self.expectedXBwLts(0,tau,k,lmb)

    def plt_downtime(self,xs=np.arange(1,100000)*0.01,lmb=0,alp=1,lmb_prob=0,t0=900.0,Y=480.0,reg='log',col='b'):
        ys = self.expected_downtime(480.0,xs=xs,lmb=lmb,reg=reg)
        ys_probs = self.prob_TgrTau(xs,lmb_prob,t0,Y)
        plt.plot(xs,(ys+ys_probs),alpha=alp,color=col)
        return (ys+ys_probs)

    def bisection(self, bisection_fn, a=1e-6, b=2000):
        n=1
        while n<10000:
            c=(a+b)/2
            if bisection_fn(c)==0 or (b-a)/2<1e-6:
                return c
            n = n + 1
            if (bisection_fn(c) > 0) == (bisection_fn(a) > 0):
                a=c
            else:
                b=c

