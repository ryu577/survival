import numpy as np
import matplotlib.pyplot as plt
from distributions.weibull import *
from distributions.lomax import *
from distributions.lognormal import *
from distributions.loglogistic import *
from nonparametric.non_parametric import *
from markovchains.markovchains import *
from datetime import datetime
from numpy import genfromtxt
import os

# k=1.1; lmb=0.5; intervention_cost=200; sample_size=5000; censor_level=200; prob=1.0
## 2.1 Comparison of non-parametric approaches when we know the distribution.
def comparisons(k=1.1, lmb=0.5, intervention_cost=200, sample_size=5000, censor_level=200, prob=1.0):
	distr_map = {'Lomax' : 1, 'Weibull' : 2, 'LogLogistic' : 3}
	l = Lomax(k=k, lmb=lmb)
	opt_tau = k*intervention_cost-1/lmb
	print("Optimal threshold: " + str(opt_tau))
	samples = l.samples(size=sample_size)
	ti, xi = censor_samples(samples, prob)
	l1 = Lomax(ti=ti, xi=xi) # for now, we have to add a dummy xi. Should be inconsequential.
	print("Original lomax kappa was:" + str(k) +\
		 "\nand for this one, " + str(l1.k) +\
		  "\nOriginal lambda was:" + str(lmb) + \
		  "\nand this one:" + str(l1.lmb))
	w1 = Weibull(ti=ti, xi=xi)
	weib_lik = w1.loglik(ti,xi,w1.k,w1.lmb)
	ll1 = LogLogistic(ti=ti, xi=xi)
	ll_lik = ll1.loglik(ti,xi,ll1.alpha,ll1.beta)
	opt_tau_l = l1.optimal_wait_threshold(intervention_cost)
	opt_tau_w = w1.optimal_wait_threshold(intervention_cost)
	opt_tau_ll = ll1.optimal_wait_threshold(intervention_cost)
	print("Based on this distribution, the optimum threshold is:" + str(opt_tau_l))
	print("Based on log logistic, the optimum threshold is:" + str(opt_tau_ll))
	costs = []
	#costs1 = []
	distr = ll1
	name = l1.__class__.__name__
	for tau in np.arange(10,900,1):
		# since there is no censoring, the distribution shouldn't matter.
		(p,t) = constr_matrices_data_distr(tau, ti=ti, xi=xi, \
			intervention_cost=intervention_cost, distr=distr)
		costs.append(time_to_absorbing(p,t,2)[0])
		#costs1.append(relative_nonparametric(samples, tau, intervention_cost))
	opt_tau_1 = np.arange(10,900,1)[np.argmin(costs)]
	print("Optimal threshold based on matrix-based non-parametric: " + str(opt_tau_1))
	#opt_tau_2 = relative_nonparametric(samples, 600.0, intervention_cost)
	opt_tau_2 = 0
	#print("Optimal threshold based on relative-savings based non-parametric: " + str(opt_tau_2))
	write_data("DataGen-Lomax_Model-LogLogistic_Censor-Det170", \
		intervention_cost, sample_size, k, lmb, opt_tau, opt_tau_ll, \
		censor_level, costs)
	return np.array([intervention_cost, sample_size,\
			 k, lmb, opt_tau, opt_tau_l, opt_tau_1, censor_level])


def censor_samples(samples, prob=1.0):
	xi = np.array([1e-3])
	unifs = np.random.uniform(size=sample_size)
	if censor_level is not None:
		ti = samples[(samples<=censor_level) + (unifs>prob)]
		if sum(samples>censor_level) > 0:
			xi = np.ones(sum((samples>censor_level) * (unifs<=prob)))*censor_level
	else:
		ti = samples
	if len(xi) == 0:
		ti = samples
		xi = np.array([1e-3])
	return ti, xi


def write_data(subfolder, intervention_cost, sample_size, k, lmb, opt_tau, \
				opt_tau_1, censor_level, costs):
	filetxt = datetime.now().year*1e10 + datetime.now().month*1e8 +\
		 datetime.now().day*1e6 + datetime.now().hour*1e4 + \
		 datetime.now().minute*1e2 + datetime.now().second
	#np.savetxt('./data/censored/lomax-' + str(filetxt) + '.csv', np.array([intervention_cost, sample_size, k, lmb, opt_tau, opt_tau_l, opt_tau_1, censor_level, name]), delimiter=',', )
	txt = str(intervention_cost)+","+str(sample_size)+"," + str(k) \
			+ "," + str(lmb) + "," + str(opt_tau) + "," \
			+ str(opt_tau_1) + "," + str(censor_level) + "\n"
	f = open('./data/' + subfolder + '/lomax-' + str(filetxt) + '.csv', 'w')
	f.write(txt)
	f.close()
	plt.plot(np.arange(10,900,1), costs)
	plt.savefig('./plots/' + str(filetxt) + '.png', bbox_inches='tight')
	plt.close()


def plot_haz_rates(l, l1, w1, ll1):
	t = np.arange(0.1,900,0.1)
	origs = l.hazard(t)
	lomaxs = l1.hazard(t)
	weibus = w1.hazard(t,w1.k,w1.lmb)
	loglgs = ll1.hazard(t)
	plt.plot(t, origs, color='g')
	plt.plot(t, lomaxs, color='r')
	plt.plot(t, weibus, color='b')
	plt.plot(t, loglgs, color='orange')

def non_parametric_comparison():
	num_samples = []
	threshs = []
	tru_opt = 0.0
	for filename in os.listdir('./data/'):
		dd = genfromtxt('./data/' + filename)
		samples = dd[1]
		num_samples.append(samples)
		tru_thresh = dd[4]
		non_prm = dd[5]
		threshs.append(non_prm)
		tru_opt = dd[4]
	plt.plot(num_samples, threshs, 'ro')
	num_samples = np.array(num_samples)
	threshs = np.array(threshs)
	stds = []
	means = []
	for i in np.unique(num_samples):
		stds.append(np.std(threshs[num_samples==i]))
		means.append(np.mean(threshs[num_samples==i]))
	means = np.array(means)
	stds = np.array(stds)
	plt.plot(np.unique(num_samples), means, 'k-')
	plt.axhline(tru_opt, color='g')
	plt.fill_between(np.unique(num_samples), means+stds, means-stds)
	plt.xlabel('Number of samples generated')
	plt.ylabel('Optimum threshold values')
	plt.show()


def test_lomax_matrix(k=1.1, lmb=0.5, intervention_cost=200):
    l = Lomax(k, lmb)
    opt_tau = k*intervention_cost-1/lmb
    print("Optimal threshold: " + str(opt_tau))
    costs = []
    probs = []
    for tau in np.arange(10,900,1):
        (p,t) = l.construct_matrices(tau, intervention_cost)
        costs.append(time_to_absorbing(p,t,2)[0])
        probs.append(steady_state(p, t)[2])
    print("Optimal thresholds based on time to absorbing state.")
    print(np.arange(10,900,1)[np.argmin(costs)])
    print("Optimal thresholds based on steady state proportions.")
    print(np.arange(10,900,1)[np.argmax(probs)])
    #plt.plot(costs)
    plt.plot(np.arange(10,900,1), probs)
    plt.show()



xs=np.arange(0.1,150,0.1)
ys=[self.loglik(self.train_org, self.train_inorg, i, ll1.beta) for i in xs]

