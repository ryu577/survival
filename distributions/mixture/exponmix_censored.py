import numpy as np
from distributions.mixture.basemixture import *
from distributions.exponential import Exponential
from scipy.stats import expon

class CensrdExpMix(BaseMix):
    def __init__(self, s, t, x, xs=None, xt=None, ws=None, wt=None, wx=None):
        self.s = s
        self.t = t
        self.x = x
        if ws is None:
            self.ws = np.ones(len(s))
        else:
            self.ws = ws
        if wt is None:
            self.wt = np.ones(len(t))
        else:
            self.wt = wt
        if wx is None:
            self.wx = np.ones(len(x))
        else:
            self.wx = wx
        if xs is None:
            self.xs = np.ones(len(s))*max(s)
        else:
            self.xs = xs
        if xt is None:
            self.xt = np.ones(len(t))*max(t)
        else:
            self.xt = xt
        self.lmb = len(self.t)/sum(self.t)
        self.mu = len(self.s)/sum(self.s)
        self.u = len(self.s)/(len(self.t)+len(self.s))

    @staticmethod
    def loglik_(mu, lmb, u, s, t, x):
        n_s = len(s)
        n_t = len(t)
        return n_s*np.log(mu)-mu*sum(s)+n_t*np.log(lmb)\
            -lmb*sum(t) + sum(np.log(u*np.exp(-mu*x)+(1-u)*np.exp(-lmb*x)))

    def loglik(self, mu=None, lmb=None, u=None):
        if mu is None:
            return self.loglik_(self.mu, self.lmb, self.u, self.s, self.t, self.x)
        else:
            return self.loglik_(mu, lmb, u, self.s, self.t, self.x)

    def loglik_prms(self, prms):
        #TODO: move to parent class
        [mu, lmb, u] = prms
        return self.loglik(mu, lmb, u)

    @staticmethod
    def grad_(mu, lmb, u, s, t, x):
        n_s = len(s)
        n_t = len(t)
        delmu = n_s/mu -sum(s) \
            - u*sum(x*np.exp(-mu*x)/(u*np.exp(-mu*x)+(1-u)*np.exp(-lmb*x)))
        dellmb = n_t/lmb -sum(t) \
            - (1-u)*sum(x*np.exp(-lmb*x)/(u*np.exp(-mu*x)+(1-u)*np.exp(-lmb*x)))
        delu = sum((np.exp(-mu*x)-np.exp(-lmb*x))/\
              (u*np.exp(-mu*x)+(1-u)*np.exp(-lmb*x)))
        return np.array([delmu, dellmb, delu])

    def grad(self, mu=None, lmb=None, u=None):
        if mu is None:
            mu=self.mu; lmb=self.lmb; u=self.u
        return CensrdExpMix.grad_(mu, lmb, u, self.s, self.t, self.x)

    def grad_prm(self, prms):
        #TODO: move to parent class
        [mu, lmb, u] = prms
        return self.grad(mu, lmb, u)

    @staticmethod
    def samples_(mu, lmb, u, n_samples, censor):
        t_len = int(n_samples*(1-u))
        s_len = int(n_samples*u)
        t_samples = np.random.exponential(1/lmb,size=t_len)
        s_samples = np.random.exponential(1/mu,size=s_len)
        if type(censor) == int or type(censor) == float:
            t_censor = np.ones(t_len)*censor
            s_censor = np.ones(s_len)*censor
        elif type(censor) == np.ndarray:
            t_censor = np.random.choice(censor, size=t_len)
            s_censor = np.random.choice(censor, size=s_len)
        x_censored = np.concatenate((t_censor[t_samples>t_censor],\
                s_censor[s_samples>s_censor]),axis=0)
        t = t_samples[t_samples<t_censor]
        s = s_samples[s_samples<s_censor]
        xt = t_censor[t_samples<t_censor]
        xs = s_censor[s_samples<s_censor]
        return s,t,x_censored, xs, xt

    def samples(self, n_samples, censor):
        return CensrdExpMix.samples_(self.mu, self.lmb, self.u\
                             ,n_samples, censor)

    @staticmethod
    def estimate_em_(s,t,x,xs,xt,ws=None,wt=None,wx=None,verbose=False):
        if ws is None:
            ws=np.ones(len(s)); wt=np.ones(len(t)); wx=np.ones(len(x))
        #ns=len(s); nt=len(t);
        ns=sum(ws); nt=sum(wt)
        #mu=len(s)/sum(s); lmb=len(t)/sum(t)
        mu=sum(ws)/sum(ws*s); lmb=sum(wt)/sum(wt*t)
        mu_prev = mu
        for tt in range(500):
            lmb_sur = np.mean(np.exp(-lmb*xt*wt))
            mu_sur = np.mean(np.exp(-mu*xs*ws))
            u = ns*(1-lmb_sur)/(ns*(1-lmb_sur)+nt*(1-mu_sur))
            tau = u*np.exp(-mu*x*wx)/(u*np.exp(-mu*x*wx)+\
                    (1-u)*np.exp(-lmb*x*wx))
            mu = sum(ws)/(sum(s*ws)+sum(tau*x*wx))
            lmb = sum(wt)/(sum(t*wt)+sum((1-tau)*x*wx))
            if verbose and tt%100 == 0:
                print("mu:" + str(mu) + ", lmb:"+str(lmb)+", u:"+str(u))
            if(abs(mu_prev-mu)/mu_prev<1e-4):
                break
            mu_prev = mu
        return mu, lmb, u

    def estimate_em(self,verbose=False):
        self.mu, self.lmb, self.u = self.estimate_em_(self.s,\
                            self.t, self.x, self.xs, self.xt, 
                            self.ws, self.wt, self.wx, verbose)    

    @staticmethod
    def u_from_lmb_mu(lmb, mu, xs, xt, ws, wt):
        ns=sum(ws); nt=sum(wt)
        lmb_sur = np.mean(np.exp(-lmb*xt*wt))
        mu_sur = np.mean(np.exp(-mu*xs*ws))
        u = ns*(1-lmb_sur)/(ns*(1-lmb_sur)+nt*(1-mu_sur))
        return u

    @staticmethod
    def u_from_lmb_mu_simplified(mu, lmb, s, t, tau):
        """
        Method u_from_lmb_mu for simple Ricks; without the hassles.
        of weights on the data, variables censoring, etc.
        (https://www.youtube.com/watch?v=CLqAFIMgpIU)
        """
        ns=sum(s); nt=sum(t)
        lmb_sur = np.exp(-lmb*tau); mu_sur = np.exp(-mu*tau)
        u = ns*(1-lmb_sur)/(ns*(1-lmb_sur)+nt*(1-mu_sur))
        return u

    @staticmethod
    def fit_censored_data(s,t,x_cen,censor):
        scale0 = Exponential.fit_censored_data(s, censor)
        scale1 = Exponential.fit_censored_data(t, censor)
        censor0_prob = 1- expon.cdf(censor, loc=0, scale=scale0)
        censor1_prob = 1 - expon.cdf(censor, loc=0, scale=scale1)
        u = (len(x_cen)/(len(s) + len(t) + len(x_cen)) - censor1_prob) / (censor0_prob - censor1_prob)
        return 1/scale0, 1/scale1, u


from distributions.lomax import Lomax

def lomax_mix():
    k1 = 1.1; lmb1 = 20
    k2 = 0.1; lmb2 = 30
    n_samples = 10000; u=0.3
    censor = 8.0

    t_len = int(n_samples*(1-u))
    s_len = int(n_samples*u)
    t_samples = Lomax.samples_(k1, lmb1, size=t_len)
    s_samples = Lomax.samples_(k2, lmb2, size=s_len)
    t = t_samples[t_samples<censor]
    s = s_samples[s_samples<censor]
    x_censored = np.ones(sum(t_samples>censor)+sum(s_samples>censor))
    
def tst_exponmix_censored_fit(size=100000):
    import random
    import matplotlib.pyplot as plt
    censor = 1.1
    lmb0 = .7
    lmb1 = 1
    u = 0.33
    track = []
    for i in range(50):
        s,t,x_cen,xs,xt = CensrdExpMix.samples_(lmb0,lmb1,u,size,censor)
        lmb0_hat, lmb1_hat, u_hat = CensrdExpMix.fit_censored_data(s,t,x_cen, censor)
        print("true lambda0 {}, lambda1 {}, mix {}; estimate lamda0 {}, lambda1 {}, mix {}".format(lmb0, lmb1, u, lmb0_hat, lmb1_hat, u_hat))
        track.append((lmb0_hat, lmb1_hat, u))

    plt.subplot(3, 1, 1)
    plt.plot(list(map(lambda x:x[0], track)))
    plt.subplot(3, 1, 2)
    plt.plot(list(map(lambda x: x[1], track)))
    plt.subplot(3, 1, 3)
    plt.plot(list(map(lambda x: x[2], track)))
    plt.title("Estimation for censored exponential mix. True params: λ1 = {}, λ2 = {}, u = {}".format(lmb0, lmb1, u))
    plt.show()



