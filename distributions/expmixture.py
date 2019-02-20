import numpy as np

class ExpMix():
    def __init__(self, s, t, x):
        self.s = s
        self.t = t
        self.x = x
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

    def loglik_prm(self, prms):
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
        return ExpMix.grad_(mu, lmb, u, self.s, self.t, self.x)

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
        return ExpMix.samples_(self.mu, self.lmb, self.u\
                             ,n_samples, censor)
    
    

def tst_em(mu_o=1/10, lmb_o=1/5, u_o=0.8, c=8):
    #mu_o=1/10; lmb_o=1/10; u_o=0.3; c=8
    #tau_o = u_o*np.exp(-mu_o*c)/(u_o*np.exp(-mu_o*c)+(1-u_o)*np.exp(-lmb_o*c))
    s, t, x, xs, xt = ExpMix.samples_(mu_o,lmb_o,u_o,50000,c)
    em = ExpMix(s,t,x)
    #mu_est = len(s)/(sum(s)+sum(tau_o*x))
    #lmb_est = len(t)/(sum(t)+sum((1-tau_o)*x))

    #print(str(mu_o)+","+str(mu_est))
    #print(str(lmb_o)+","+str(lmb_est))
    ns=len(s); nt=len(t); nx=len(x)

    #u_est = ns*(1-np.exp(-lmb_o*c))/(ns*(1-np.exp(-lmb_o*c))+nt*(1-np.exp(-mu_o*c)))
    #print(str(u_o)+","+str(u_est))    

    lmb=0.2; mu=0.9
    for tt in range(500):
        lmb_sur = np.mean(np.exp(-lmb*xt))
        mu_sur = np.mean(np.exp(-mu*xs))
        u = ns*(1-lmb_sur)/(ns*(1-lmb_sur)+nt*(1-mu_sur))
        ## The actual probability of seeing sample beyong censor pt from sample-1.
        tau = u*np.exp(-mu*x)/(u*np.exp(-mu*x)+(1-u)*np.exp(-lmb*x))
        ## Use tau to estimate the rate parameters.
        mu = len(s)/(sum(s)+sum(tau*x))
        lmb = len(t)/(sum(t)+sum((1-tau)*x))
        if tt%100 == 0:
            print("mu:" + str(mu) + ", lmb:"+str(lmb)+", u:"+str(u))


if __name__ == '__main__':
    mu = 1/10; lmb=1/5; u=0.5;
    s,t,x = ExpMix.samples_(mu,lmb,u,50000,8)
    em = ExpMix(s,t,x)
    grd = em.grad(mu,lmb,u)
    mu_est = len(s)/(sum(s) + u*sum(x))
    lmb_est = len(t)/(sum(t) + (1-u)*sum(x))
    print(mu-mu_est)
    print(lmb-lmb_est)
    print(grd)




