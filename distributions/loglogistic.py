import numpy as np
from misc.sigmoid import *
from distributions.basemodel import *
from distributions.lomax import *
from distributions.weibull import *
import pandas as pd

class LogLogistic(Base):
    '''
    The log logistic distribution: 
    https://en.wikipedia.org/wiki/Log-logistic_distribution
    Since we have alpha as the shape parameter and beta as the scale parameter
    in the link above while k for shape and lmb for scale more generally,
    the instance of this distribution will have alpha=k always
    and beta=lmb always.
    '''
    ## Coefficients for linear regression obtaining LogLogistic parameters
    ## from Weibull and Lomax parameters.
    lin_alphas = np.array([ -2.52772215e+00,  -2.65188344e+00,  -1.04822376e-01,
                                    1.30662288e+01,   2.41382642e-01,   6.66425463e-03,
                                    -4.77346745e-04,   2.54565486e-01,   5.85225529e-04,
                                    -4.26451352e-01,  -2.85319115e-01,   3.00439683e-02,
                                    -2.59997264e-02,   2.70681101e-03,  -1.12247885e-02])
    lin_betas = np.array([ -1.50974077e-02,   8.57431339e-02,   1.15325271e-03,
                                    1.80669587e+00,   1.73089858e-04,  -8.17185130e-05,
                                    -1.02613704e-07,  -5.69980568e-03,   5.46670938e-06,
                                    -1.33985957e-02,  -4.05188576e-03,   1.20595620e-03,
                                    9.47605555e-06,  -1.73386464e-06,  -1.79663515e-03])

    def __init__(self, ti=None, xi=None, beta=0.5,alp=1,
                 params=np.array([1.05, 200.1]),
                 w_org=None, w_inorg=None, verbose=False,
                 step_lengths=np.array([1e-8,1e-5,1e-3,1e-2,1e-1,1.0,3.0,5.0,10.0])):
        '''
        Initializes an instance of the log logistic distribution.
        '''
        if ti is not None:
            self.train_org = ti
            self.train_inorg = xi
            # These weights need to be manually set to something
            # different if desired. For example, per feature.
            if w_org is None:
                self.w_org = np.ones(len(ti))
            else:
                self.w_org = w_org
            if w_inorg is None:
                self.w_inorg = np.ones(len(xi))
            else:
                self.w_inorg = w_inorg
            self.gradient_descent(params=params, verbose=verbose,
                            step_lengths=step_lengths)
        else:
            self.train = []
            self.test = []
            self.train_org = self.samples_(alp,beta)
            self.train_inorg = np.array([.01])
            self.w_org = np.ones(len(self.train_org))
            self.w_inorg = np.ones(len(self.train_org))
            self.alpha = self.lmb = alp
            self.beta = self.k = beta
            self.params = []
        # Pre-computed linear regression coefficients for fast-logistic regression.
        self.lin_alphas = np.array([ -2.52772215e+00,  -2.65188344e+00,  -1.04822376e-01,
                                    1.30662288e+01,   2.41382642e-01,   6.66425463e-03,
                                    -4.77346745e-04,   2.54565486e-01,   5.85225529e-04,
                                    -4.26451352e-01,  -2.85319115e-01,   3.00439683e-02,
                                    -2.59997264e-02,   2.70681101e-03,  -1.12247885e-02])
        self.lin_betas = np.array([ -1.50974077e-02,   8.57431339e-02,   1.15325271e-03,
                                    1.80669587e+00,   1.73089858e-04,  -8.17185130e-05,
                                    -1.02613704e-07,  -5.69980568e-03,   5.46670938e-06,
                                    -1.33985957e-02,  -4.05188576e-03,   1.20595620e-03,
                                    9.47605555e-06,  -1.73386464e-06,  -1.79663515e-03])

    def set_params(self, beta, alpha, params=None):
        '''
        Sets the parameters. Need a saperate method
        for this distribution since it has alpha
        and k that mean the same thing and beta and 
        lmb that mean the same thing.
        args:
            alpha: The shape parameter.
            beta: The scale parameter.
            params: An array of shape and scale parameters.
        '''
        if params is not None:
            [beta, alpha] = params[:2]
        self.k = self.beta = beta
        self.lmb = self.alpha = alpha
        self.params = [beta, alpha]

    def determine_params(self, k, lmb, params):
        '''
        Determines the parameters. Defined in basemodel.py
        '''
        return super(LogLogistic, self).determine_params(k, lmb, params)

    @staticmethod
    def pdf_(x, beta, alpha):
        '''
        Returns the probability density function of the distribution.
        args:
            x: The value at which the PDF is to be calculated.
            alpha: The shape paramter.
            beta: The scale parameter.
        '''
        return (beta / alpha) * (x / alpha)**(beta - 1) / \
                (1 + (x / alpha)**beta)**2

    def pdf(self, x):
        return LogLogistic.pdf_(x, self.beta, self.alpha)

    @staticmethod
    def cdf_(x, beta, alpha):
        '''
        The cumulative density function.
        args:
            x: The value at which the CDF is to be calculated.
            alpha: The shape parameter of the distribution.
            beta: The scale parameter of the distribution.
        '''
        return 1 / (1 + (x / alpha)**-beta)

    def cdf(self, x):
        return LogLogistic.cdf_(x, self.beta, self.alpha)

    @staticmethod
    def inv_cdf(self, u):
        '''
        The inverse CDF of the dsitribution (used for generating random samples).
        args:
            u: A number between 0 and 1.
            alpha: The shape parameter.
            beta: The scale parameter.
        '''
        return LogLogistic.inv_cdf_(u, self.beta, self.alpha)

    @staticmethod
    def inv_cdf_(u, beta, alpha):
        return alpha * (1 / u - 1)**(-1 / beta)

    def samples(self, beta=None, alpha=None, size=1000):
        '''
        Generates samples from a log logistic distribution.
        args:
            size: The number of samples to be generated.
            alpha: The shape parameter of the distribution.
            beta: The scale parameter of the distribution.
        '''
        return LogLogistic.inv_cdf_(np.random.uniform(size=size), self.beta, self.alpha)

    @staticmethod
    def samples_(beta, alpha, size=1000):
        return LogLogistic.inv_cdf_(np.random.uniform(size=size), beta, alpha)

    @staticmethod
    def logpdf_(x, beta, alpha):
        '''
        The logarithm of the PDF of the distribution.
        args:
            x: The value at which the function is to be evaluated.
            alpha: The shape parameter.
            beta: The scale parameter.
        '''
        return np.log(beta) - np.log(alpha) +\
            (beta - 1) * (np.log(x) - np.log(alpha)) \
            - 2 * np.log(1 + (x / alpha)**beta)

    def logpdf(self, x):
        return LogLogistic.logpdf_(x, self.beta, self.alpha)

    @staticmethod
    def survival_(x, beta, alpha):
        return 1/(1+(x/alpha)**beta)
    
    def survival(self, x):
        '''
        The survival function of the distribution 
        (probability that it is greater than x).
        args:
            x: Evaluated here.
            alpha: Shape parameter.
            beta: Scale parameter.
        '''
        return LogLogistic.survival_(x, self.beta, self.alpha)

    @staticmethod
    def logsurvival_(x, beta, alpha):
        '''
        The logarithm of the survival function of the
        distribution (probability that it is greater than x).
        For now, we simply take the log, but we can possibly
        improve the efficiency numerically in the future.
        args:
            x: Evaluated here.
            alpha: Shape parameter.
            beta: Scale parameter.
        '''
        return np.log(LogLogistic.survival_(x, beta, alpha))

    def logsurvival(self, x):
        return LogLogistic.logsurvival_(x, self.beta, self.alpha)

    @staticmethod
    def loglik_(t, x, w_org, w_inorg, beta, alpha):
        '''
        The log likelihood of the loglogistic distribution.
        args:
            t: The array of observed survival times.
            x: The array of censored survival times.
            alpha: The shape parameter.
            beta: The scale parameter.
        '''
        if len(w_org) == len(t) and len(w_inorg) == len(x):
            return sum(w_org * LogLogistic.logpdf_(t, beta, alpha)) + \
                sum(w_inorg * LogLogistic.logsurvival_(x, beta, alpha))
        else:
            return sum(LogLogistic.logpdf_(t, beta, alpha)) +\
                sum(LogLogistic.logsurvival_(x, beta, alpha))

    def loglik(self, t=None, x=None, beta=None, alpha=None):
        if t is None:
            t = self.train_org
        if x is None:
            x = self.train_inorg
        if beta is None:
            beta = self.beta
        if alpha is None:
            alpha=self.alpha
        return LogLogistic.loglik_(t, x, self.w_org, self.w_inorg,
                    beta, alpha)
    
    @staticmethod
    def grad_(t, x, w_org, w_inorg, beta, alp):
        '''
        Analytically calculates the gradient.
        args:
            t: The array of observed survival times.
            x: The array of censored survival times.
            alp: The scale parameter.
            beta: The shape parameter.
        '''
        if len(w_org) == len(t) and len(w_inorg) == len(x):
            n = np.sum(w_org)
            m = np.sum(w_inorg)
            delalp = -n * beta / alp + 2 * beta / alp**(beta + 1) \
                *sum(t**beta / (1 + (t / alp)**beta) * w_org) \
                + beta / alp**(beta + 1) * sum(x**beta /
                                               (1 + (x / alp)**beta) * w_inorg)
            delbeta = n / beta - n * np.log(alp) + sum(np.log(t) * w_org)\
                - 2 * sum((t / alp)**beta / (1 + (t / alp)**beta)
                          * np.log(t / alp) * w_org) \
                - sum((x / alp)**beta / (1 + (x / alp)**beta)
                      * np.log(x / alp) * w_inorg)
        else:
            n = len(t)
            m = len(x)
            delalp = -n * beta / alp + 2 * beta / alp**(beta + 1) *\
             sum(t**beta / (1 + (t / alp)**beta)) \
                + beta / alp**(beta + 1) * sum(x**beta / (1 + (x/alp)**beta))
            delbeta = n / beta - n * np.log(alp) + \
                sum(np.log(t)) - 2 * sum((t/alp)**beta / (1 + (t/alp)**beta) * np.log(t/alp)) \
                - sum((x / alp)**beta / (1 + (x / alp)**beta) * np.log(x/alp))
        return np.array([delbeta, delalp])

    def grad(self, t=None,x=None,beta=None,alpha=None):
        if t is None:
            t = self.train_org
        if x is None:
            x = self.train_inorg
        if beta is None:
            beta = self.beta
        if alpha is None:
            alpha = self.alpha
        return LogLogistic.grad_(t,x,self.w_org,self.w_inorg, beta, alpha)

    @staticmethod
    def grad_l_pdf_(t, beta, alpha):
        tmp = (x/alpha)**beta
        delbeta = -tmp / (1 + tmp) * np.log(x / alpha)
        delalpha = beta / alpha * tmp / (1 + tmp)
        return np.array([delbeta, delalpha])

    def grad_l_pdf(self, t):
        return LogLogistic.grad_l_pdf_(t, self.beta, self.alpha)

    @staticmethod
    def grad_l_survival_(x, beta, alpha):
        temp = (x / alpha)**beta
        delbeta = -temp / (1 + temp) * np.log(x / alpha)
        delalpha = beta / alpha * temp / (1 + temp)
        return np.array([delbeta, delalpha])

    def grad_l_survival(self, x):
        return LogLogistic.grad_l_survival_(x, self.beta, self.alpha)

    def hessian(self, t, x, k=0.5, lmb=0.3):
        '''
        TODO: Calculate the hessian matrix of the log likelihood function
        instead of defaulting to the numerical version.
        args:
            t: The array of observed survival times.
            x: The array of censored survival times.
            k: The shape parameter.
            lmb: The scale parameter.
        '''
        return self.numerical_hessian(t, x, k, lmb)

    @staticmethod
    def ll_haz_rate_(beta, alpha, t):
        return (beta/alpha)*(t/alpha)**(beta-1)/(1+(t/alpha)**beta)

    def hazard(self, t):
        return self.ll_haz_rate_(self.beta, self.alpha, t)

    @staticmethod
    def mean_(beta, alpha):
        return alpha*np.pi/(beta*np.sin(np.pi/beta))

    @staticmethod
    def train_fast_():
        '''
        Fast LogLogistic estimation, based on inferring
        coefficients using Lomax and Weibull coefficients.
        '''
        train_df = pd.DataFrame()
        # Generate some training data at random.
        for _ in range(100):
            k = np.random.uniform()*2.0
            lmb = np.random.uniform()*20.0
            ti = LogLogistic.samples_(lmb,k)
            lmx = Lomax.est_params(ti)
            wbl = Weibull.est_params(ti)
            train_df = train_df.append({'alpha':lmb,'beta':k,\
                            'lomax_k':lmx[0],'lomax_lmb':lmx[1],
                            'weib_k':wbl[0],'weib_lmb':wbl[1]},
                            ignore_index=True)
        train_df["weib_lmb"][np.isinf(train_df["weib_lmb"])] = 1000.0
        train_df["weib_lmb"][train_df["weib_lmb"]>1000.0] = 1000.0
        train_df["lomax_k"][np.isnan(train_df["lomax_k"])]=10.0
        x_features = np.ones((len(train_df),15))
        x_features[:,1] = train_df["lomax_k"]
        x_features[:,2] = train_df["lomax_lmb"]
        x_features[:,3] = train_df["weib_k"]
        x_features[:,4] = train_df["weib_lmb"]
        x_features[:,5] = train_df["weib_k"]**2
        x_features[:,6] = train_df["weib_lmb"]**2
        x_features[:,7] = train_df["lomax_k"]**2
        x_features[:,8] = train_df["lomax_lmb"]**2
        x_features[:,9] = train_df["lomax_k"]*train_df["lomax_lmb"]
        x_features[:,10] = train_df["lomax_k"]*train_df["weib_k"]
        x_features[:,11] = train_df["lomax_k"]*train_df["weib_lmb"]
        x_features[:,12] = train_df["lomax_lmb"]*train_df["weib_k"]
        x_features[:,13] = train_df["lomax_lmb"]*train_df["weib_lmb"]
        x_features[:,14] = train_df["weib_k"]*train_df["weib_lmb"]
        y_alp, lin_alphas = predicn(train_df, x_features)
        y_beta, lin_betas = predicn(train_df, x_features, "beta")
        return lin_alphas, lin_betas, y_alp, y_beta, train_df

    @staticmethod
    def est_params_fast_(ti,xi,lin_betas=None, lin_alphas=None):
        if lin_alphas is None:
            lin_alphas = LogLogistic.lin_alphas
        if lin_betas is None:
            lin_betas = LogLogistic.lin_betas
        ftrs = cnstrct_feature(ti, xi)
        alpha = sum(ftrs*lin_alphas)
        beta = sum(ftrs*lin_betas)
        return beta, alpha

def predicn(train_df, x_features, trm="alpha"):
    y_alp = train_df[trm]
    X = x_features
    lhs = np.dot(X.T,X)
    rhs = np.dot(X.T,y_alp)
    betas = np.linalg.solve(lhs,rhs)
    y_pred = np.dot(X,betas)
    return y_pred, betas


def cnstrct_feature(ti, xi=None):
    lmx = Lomax.est_params(ti)
    wbl = Weibull.est_params(ti)
    x_features = np.array([1,lmx[0],lmx[1],wbl[0],wbl[1],
                          wbl[0]**2,wbl[1]**2,lmx[0]**2,lmx[1]**2,
                          lmx[0]*lmx[1],lmx[0]*wbl[0],lmx[0]*wbl[1],
                          lmx[1]*wbl[0],lmx[1]*wbl[1],wbl[0]*wbl[1]])
    return x_features


