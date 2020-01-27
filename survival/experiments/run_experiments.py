import numpy as np
import matplotlib.pyplot as plt
from survival.distributions.weibull import *
from survival.distributions.lomax import *
from survival.distributions.lognormal import *
from survival.distributions.loglogistic import *
from survival.nonparametric.non_parametric import *
from survival.markovchains.markovchains import *
from datetime import datetime
from numpy import genfromtxt
import os


#k=1.1; lmb=0.5; intervention_cost=200; q=1.0
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
    props = []
    for tau in np.arange(10,900,1):
        (p,t) = l.construct_matrices(tau, intervention_cost)
        p1, t1 = expand_matrices(p,t,q)
        ## Verified that as long as  the row corresponding to the Ready state
        ## sums to 1, the optimal threshold doesn't change.
        costs.append(time_to_absorbing(p,t,2)[0])
        steady = steady_state(p1, t1)
        props.append(steady[0]+steady[3])
        #props.append(steady_state(p, t)[2])
    print("Optimal thresholds based on time to absorbing state.")
    print(np.arange(10,900,1)[np.argmin(costs)])
    print("Optimal thresholds based on steady state proportions.")
    print(np.arange(10,900,1)[np.argmax(props)])
    plot_side_by_side(props, costs, np.arange(10,900,1))


def expand_matrices(p, t, q=1.0):
    p1 = np.zeros((4,4))
    t1 = np.zeros((4,4))
    p1[0,2] = p[0,1]
    p1[0,3] = p[0,2]
    t1[0,2] = t[0,1]
    t1[0,3] = t[0,2]
    
    p1[1,2] = p[0,1]
    p1[1,3] = p[0,2]
    t1[1,2] = t[0,1]
    t1[1,3] = t[0,2]

    p1[2,3] = 1.0
    t1[2,3] = t[1,2]
    
    p1[3,0] = (1-q)
    p1[3,1] = q
    t1[3,0] = 10000.0
    t1[3,1] = 10000.0 #Any arbitrary number.
    return np.matrix(p1), np.matrix(t1)


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



