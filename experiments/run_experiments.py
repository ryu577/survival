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


def optimal_time_same_for_steadystate_and_timetoabsorbing(k=1.1, lmb=0.5, intervention_cost=200, q=1.0):
    """
    Demonstrates that the threshold that optimizes time required to 
    go from Unhealthy to Ready is also the threshold that optimizes
    steady state probability of being in Ready.
    args:
        q: The static probability that the VMs on the node will be down.
    """
    l = Lomax(k, lmb)
    opt_tau = k*intervention_cost/q-1/lmb
    print("Optimal threshold: " + str(opt_tau))
    costs = []
    probs = []
    for tau in np.arange(10,900,1):
        (p,t) = l.construct_matrices(tau, intervention_cost)
        ## Verified that as long as  the row corresponding to the Ready state
        ## sums to 1, the optimal threshold doesn't change.
        prob_rdy_to_powon = (1-q)*l.survival(tau)
        p[2,] = np.matrix([1-prob_rdy_to_powon, prob_rdy_to_powon, 0])
        t[2,] = np.matrix([2.0, tau, 0])
        costs.append(time_to_absorbing(p,t,2)[0])
        probs.append(steady_state(p, t)[2])
    print("Optimal thresholds based on time to absorbing state.")
    print(np.arange(10,900,1)[np.argmin(costs)])
    print("Optimal thresholds based on steady state proportions.")
    print(np.arange(10,900,1)[np.argmax(probs)])
    plot_side_by_side(probs, costs, np.arange(10,900,1))


def plot_side_by_side(ser1, ser2, t=np.arange(10,900,1)):
    """
    See: https://matplotlib.org/examples/api/two_scales.html
    """
    fig, ax1 = plt.subplots()
    ax1.plot(t, ser1, 'b-')
    ax2 = ax1.twinx()
    ax2.plot(t, ser2, 'b-', color='r')
    #plt.plot(costs)
    #plt.plot(np.arange(10,900,1), probs)
    plt.show()


## Demonstration that weighted log likelihood model is working.

k=1.1; lmb=0.5; intervention_cost=200; sample_size=5000; censor_level=200; prob=1.0
l = Lomax(k=k, lmb=lmb)
samples = l.samples(size=sample_size)
ti, xi = censor_samples(samples, prob)
w_org = np.ones(len(ti))
w_org[:20] = 4.5
w_inorg = np.ones(len(xi))
w_inorg[:20] = 3.5
ll1 = LogLogistic(ti=ti, xi=xi, w_org=w_org, w_inorg=w_inorg)

ll1.numerical_grad(ti,xi,1.7,1.1)
#array([ -24.62953062, -398.78820453])

ll1.grad(ti,xi,1.7,1.1)
#array([ -24.62953032, -398.78820572])

ll1.loglik(ti,xi,1.7,1.1)
#-12115.123151118494

ll1.w_org[:3] = 2.0
ll1.w_inorg[:10] = 3.5

ll1.loglik(ti,xi,1.7,1.1)
#-12259.249396962305

ll1.numerical_grad(ti,xi,1.7,1.1)
# array([  -7.50088857, -518.54139128])

ll1.grad(ti,xi,1.7,1.1)
#array([  -7.50088825, -518.54139243])



