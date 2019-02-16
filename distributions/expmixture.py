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
            return ExpMix.loglik_(self.mu, self.lmb, self.u, self.s, self.t, self.x)            
        else:
            return ExpMix.loglik_(mu, lmb, u, self.s, self.t, self.x)

    @staticmethod
    def grad_(mu, lmb, u, s, t, x):
        n_s = len(s)
        n_t = len(t)
        delmu = n_s/mu -sum(s) \
            - u*mu*sum(np.exp(-mu*x)/(u*np.exp(-mu*x)+(1-u)*np.exp(-lmb*x)))
        dellmb = n_t/lmb -sum(t) \
            - (1-u)*lmb*sum(np.exp(-lmb*x)/(u*np.exp(-mu*x)+(1-u)*np.exp(-lmb*x)))
        delu = sum((np.exp(-mu*x)-np.exp(-lmb*x))/\
              (u*np.exp(-mu*x)+(1-u)*np.exp(-lmb*x)))
        return np.array([delmu, dellmb, delu])

    def grad(self, mu=None, lmb=None, u=None):
        if mu is None:
            mu=self.mu; lmb=self.lmb; u=self.u
        return ExpMix.grad_(mu, lmb, u, self.s, self.t, self.x)

    @staticmethod
    def samples_(mu, lmb, u, n_samples, censor):
        t_len = int(n_samples*(1-u))
        s_len = int(n_samples*u)
        t_samples = np.random.exponential(1/lmb,size=t_len)
        s_samples = np.random.exponential(1/mu,size=s_len)
        x = np.concatenate((t_samples[t_samples>censor],\
                s_samples[s_samples>censor]),axis=0)
        t = t_samples[t_samples<censor]
        s = s_samples[s_samples<censor]
        return s,t,x

    def samples(self, n_samples, censor):
        return ExpMix.samples_(self.mu, self.lmb, self.u\
                             ,n_samples, censor)


if __name__ == '__main__':
    s,t,x = ExpMix.samples_(1/10,1/7,0.3,3000,8)
    em = ExpMix(s,t,x)
    em.grad(1/10,1/7,0.3)


