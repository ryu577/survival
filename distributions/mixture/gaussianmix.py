import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from itertools import groupby
import matplotlib.pyplot as plt

class GaussMix():    
    def __init__(self,mu1,sigma1,mu2,sigma2,p):
        self.mu1=mu1; self.sigma1=sigma1; self.mu2=mu2
        self.sigma2=sigma2; self.p=p
    
    @staticmethod
    def samples_(mu1, sigma1, mu2, sigma2, p, n_samples=5000):
        sampl_1_len = int(n_samples*p)
        sampl_2_len = int(n_samples*(1-p))
        x1 = np.random.normal(mu1,sigma1,sampl_1_len)
        x2 = np.random.normal(mu2,sigma2,sampl_2_len)
        x = np.concatenate((x1,x2),axis=0)
        return x

    def samples(self, n_samples=5000):
        mu1, mu2, sigma1, sigma2, p = self.mu1, self.mu2, self.sigma1, self.sigma2, self.p

        return GaussMix.samples_(mu1, sigma1, mu2, sigma2, p, n_samples)

    @staticmethod
    def loglik_(x, mu1, sigma1, mu2, sigma2, p):
        return sum(np.log(norm.pdf(x, mu1, sigma1) * p + norm.pdf(x, mu2, sigma2) * (1 - p)))

    def loglik(self,x):
        mu1,mu2,sigma1,sigma2,p=self.mu1,self.mu2,self.sigma1,self.sigma2,self.p
        return GaussMix.loglik_(x,mu1,mu2,sigma1,sigma2,p)
    
    def loglik_p(self, x, prms):
        """
        A log likelihood function that takes a params vector
        as input.
        """
        mu1,sigma1,mu2,sigma2,p = prms
        return GaussMix.loglik_(x, mu1,sigma1,mu2,sigma2,p)

    @staticmethod
    def grad_(x,mu1, sigma1, mu2, sigma2, p):
        delp_numr = norm.pdf(x,mu1,sigma1)-norm.pdf(x,mu2,sigma2)
        delp_denom = norm.pdf(x,mu1,sigma1)*p+norm.pdf(x,mu2,sigma2)*(1-p)
        delp_trms = delp_numr/delp_denom
        delp = sum(delp_numr/delp_denom)
        delmu1_numr = p*(x-mu1)/sigma1**2*norm.pdf(x,mu1,sigma1)
        delmu1_trms = delmu1_numr/delp_denom
        delmu1 = sum(delmu1_trms)
        delmu2_numr = (1-p)*(x-mu2)/sigma2**2*norm.pdf(x,mu2,sigma2)
        delmu2_trms = delmu2_numr/delp_denom
        delmu2 = sum(delmu2_trms)
        delsigma1_numr = p*norm.pdf(x,mu1,sigma1)*((x-mu1)**2/sigma1**3-1/sigma1)
        delsigma1_trms = delsigma1_numr/delp_denom
        delsigma1 = sum(delsigma1_trms)
        delsigma2_numr = (1-p)*norm.pdf(x,mu2,sigma2)*((x-mu2)**2/sigma2**3-1/sigma2)
        delsigma2_trms = delsigma2_numr/delp_denom
        delsigma2 = sum(delsigma2_trms)
        return np.array([delmu1,delsigma1,delmu2,delsigma2,delp])

    def grad(self, x):
        mu1,mu2,sigma1,sigma2,p=self.mu1,self.mu2,self.sigma1,self.sigma2,self.p
        return GaussMix.grad_(x,mu1,mu2,sigma1,sigma2,p)

    @staticmethod
    def numr_grad_(x,mu1,sigma1,mu2,sigma2,p):
        eps = 1e-5
        prms = np.array([mu1,sigma1,mu2,sigma2,p])
        grd = np.zeros(5)
        for i in range(5):
            prms[i] -= eps
            [mu1, sigma1, mu2, sigma2, p] = prms
            lik1 = GaussMix.loglik_(x, mu1, sigma1, mu2, sigma2, p)
            prms[i] += eps
            [mu1, sigma1, mu2, sigma2, p] = prms
            lik2 = GaussMix.loglik_(x, mu1, sigma1, mu2, sigma2, p)
            grd[i] = (lik2 - lik1) / 2 / eps

            prms[i] += eps
        return grd

    def numr_grad(self, x):
        mu1, mu2, sigma1, sigma2, p = self.mu1, self.mu2, self.sigma1, self.sigma2, self.p
        return GaussMix.numr_grad_(x, mu1, sigma1, mu2, sigma2, p)

    def numr_fit(self,x, n_component):
        kmeans = KMeans(n_clusters=n_component).fit(x.reshape(-1, 1))
        mu1, mu2 = kmeans.cluster_centers_
        mu1, mu2 = mu1[0], mu2[0]
        lens = [len(list(group)) for key, group in groupby(sorted(kmeans.labels_))]
        p = lens[0]/(float)(sum(lens))
        sigma1 = np.std(x[kmeans.labels_==0]-mu1)
        sigma2 = np.std(x[kmeans.labels_==1]-mu2)
        params_init = (mu1,sigma1, mu2, sigma2, p)
        liklihood_init = GaussMix.loglik_(x, mu1, sigma1, mu2, sigma2, p)
        print("init mu1,sig1,mu2, sig2, p: {}".format((mu1,sigma1, mu2, sigma2, p)))
        lr = 1e-2 #Learning Rate.
        max_iter = 10000
        iter = 0
        n_data = len(x)
        track = []
        last_liklihood = -1e9
        delta_likelhood = 1
        while (iter < max_iter and delta_likelhood>0):
            liklihood = GaussMix.loglik_(x, mu1, sigma1, mu2, sigma2, p)
            print("iter: {}, liklihood {}, mu1,sig1,mu2, sig2, p {}".format(iter, liklihood, (mu1,sigma1,mu2, sigma2, p)))
            grd = GaussMix.numr_grad_(x, mu1, sigma1, mu2, sigma2, p)
            [mu1, sigma1, mu2, sigma2, p] = [mu1, sigma1, mu2, sigma2, p] + lr * grd / n_data
            delta_likelhood = liklihood - last_liklihood
            iter += 1
            last_liklihood = liklihood
            track.append(liklihood)
        plt.plot(track)
        plt.title("Liklihood Increase trend")
        print("Fit result after {} iteration: ".format(iter))
        print("Init likelihood {}, Param (mu1, sigma1, mu2, sigma2, p): {}".format(liklihood_init, params_init))
        print("Final likelihood {}, Param (mu1, sigma1, mu2, sigma2, p): {}".format(track[-1], (mu1, sigma1, mu2, sigma2, p)))
        plt.show()
