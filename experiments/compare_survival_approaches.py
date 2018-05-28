import matplotlib.pyplot as plt
from survival.weibull import *
from survival.lomax import *
from survival.lognormal import *
from survival.loglogistic import *
from survival.non_parametric import *
from markovchains.markovchains import *
from datetime import datetime
from numpy import genfromtxt
import os
import matplotlib.pyplot as plt

## 2.1 Comparison of non-parametric approaches when we know the distribution.

def comparisons(k=1.1, lmb=0.5, intervention_cost=200, sample_size=1000):
	l = Lomax(k=k, lmb=lmb)
	opt_tau = k*intervention_cost-1/lmb
	print("Optimal threshold: " + str(opt_tau))
	samples = l.samples(size=sample_size)
	l1 = Lomax(ti=samples, xi=np.array([1e-3])) # for now, we have to add a dummy xi. Should be inconsequential.
	w1 = Weibull(ti=samples, xi=np.array([1e-3]))
	ll1 = LogLogistic(ti=samples, xi=np.array([1e-3]))
	print("Original lomax kappa was:" + str(k) + "\nand for this one, " + str(l1.k) + "\nOriginal lambda was:" + str(lmb) + "\nand this one:" + str(l1.lmb))
	opt_tau_0 = l1.optimalWaitThreshold(200)
	opt_tau_w = w1.optimalWaitThreshold(200)
	#opt_tau_ll = ll1.optimalWaitThreshold(200)
	print("Based on this distribution, the optimum threshold is:" + str(opt_tau_0))
	costs = []
	#costs1 = []
	for tau in np.arange(10,600,1):
		# since there is no censoring, the data shouldn't matter.
		(p,t) = constr_matrices_data_distr(tau, ti=samples, intervention_cost=intervention_cost)
		costs.append(time_to_absorbing(p,t,2)[0])
		#costs1.append(relative_nonparametric(samples, tau, intervention_cost))
	opt_tau_1 = np.arange(10,600,1)[np.argmin(costs)]
	print("Optimal threshold based on matrix-based non-parametric: " + str(opt_tau_1))
	opt_tau_2 = relative_nonparametric(samples, 600.0, intervention_cost)
	print("Optimal threshold based on relative-savings based non-parametric: " + str(opt_tau_2))
	filetxt = datetime.now().year*1e10 + datetime.now().month*1e8 + datetime.now().day*1e6 + datetime.now().hour*1e4 + datetime.now().minute*1e2 + datetime.now().second
	np.savetxt('./data/lomax-' + str(filetxt) + '.csv', np.array([intervention_cost, sample_size, k, lmb, opt_tau, opt_tau_0, opt_tau_1, opt_tau_2]), delimiter=',', )
	return np.array([intervention_cost, sample_size, k, lmb, opt_tau, opt_tau_0, opt_tau_1, opt_tau_2])


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


