from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from functools import partial

#state_indices = ["Raw","Ready","Unhealthy","Booting","PoweringOn","Dead","HumanInvestigate","Recovering"]
def time_to_absorbing(
        p = np.matrix([
              [0,.2,.4,.3,.1,0,0],
              [.2,0,.3,.4,0,.1,0],
              [.1,.3,0,.5,0,.1,0],
              [.2,.3,.2,0,.1,.1,.1],
              [.2,.2,0,.5,0,.1,0],
              [.2,.3,0,.1,0,0,.4],
              [.2,.1,.2,.3,0,.2,0]
              ]),
        t = np.matrix([
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2]
              ]),
        absorbing_state = 1
              ):
    rhs = np.diag(np.dot(p, t.T))
    rhs = np.delete(rhs,absorbing_state)
    q = np.delete(p,np.s_[absorbing_state],1)
    q = np.delete(q,np.s_[absorbing_state],0)
    lhs = (np.eye(q.shape[0]) - q)
    x = np.linalg.solve(lhs,rhs)
    return np.insert(x,absorbing_state,0)


def absorbingstatemontecarlo(
        p = np.matrix([
              [0,.2,.4,.4],
              [.2,0,.4,.4],
              [.2,.3,0,.5],
              [.3,.4,.3,0]
              ]),
        t = np.matrix([
              [1,2,3,1],
              [5,1,7,1],
              [1,1,1,1],
              [1,1,1,1]
              ]),
        starting_state = 2,
        absorbing_state = 3
              ):
    sum_times = 0
    states = {}
    for i in range(10000):
        curr_state = starting_state
        hist = str(starting_state)
        while curr_state != absorbing_state:
            #next_state = np.random.choice(p.shape[0], p=np.array(p[curr_state,])[0])
            next_state = np.random.choice(p.shape[0], p=np.array(p[curr_state,]))
            sum_times = sum_times + t[curr_state,next_state]
            curr_state = next_state
            hist = hist + "," + str(curr_state)
        if hist in states:
            states[hist] = states[hist] + 1
        else:
            states[hist] = 1
    print("Simulation: " + str(sum_times/10000))
    return states


def steadystatemontecarlo(
        p = np.matrix([
              [0,.2,.4,.4],
              [.2,0,.4,.4],
              [.2,.3,0,.5],
              [.3,.4,.3,0]
              ]),
        t = np.matrix([
              [2,2,2,1],
              [1,1,5,1],
              [1,1,1,1],
              [1,1,1,1]
              ]),
        starting_state = 2 #Where to start simulation, shouldn't matter.
  ):
    '''
    Given the transition probabilities and transition times matrices, outputs the
    proportion of time spent in each state via simulation and closed form.
    The two should always be close.
    '''
    states = np.zeros(p.shape[0])
    states_w_times = np.zeros(p.shape[0])
    curr_state = starting_state #Shouldn't matter.
    for i in range(10000):
      next_state = np.random.choice(p.shape[0], p=np.array(p[curr_state,])[0])
      states[curr_state] = states[curr_state] + 1
      states_w_times[curr_state] = states_w_times[curr_state] + t[curr_state, next_state]
      curr_state = next_state
    p_times_t = np.array(np.sum(np.multiply(p,t),axis=1).T)[0]
    # Solve for pi by finding the null space of (P-I)
    pis = np.linalg.svd(p-np.eye(p.shape[0]))[0][:,p.shape[0]-1]
    pis = pis/sum(pis)
    res = [states_w_times/sum(states_w_times), pis, p_times_t]
    props1 = np.multiply(res[1].T,res[2])/sum(np.multiply(res[1].T,res[2]).T)
    props2 = res[0]
    return [props1, props2]


def steady_state_props(
  p=np.matrix([
              [0,.2,.4,.4],
              [.2,0,.4,.4],
              [.2,.3,0,.5],
              [.3,.4,.3,0]
              ])):
    '''
    Calculates the proportion time spend in each state
    in steady state.
    Based on the method outlined in the answer here:
    https://math.stackexchange.com/a/2452452/155881
    args:
      p: The matrix or 2d array with transition probabilities.      
    '''
    dim = p.shape[0]
    q = (p-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    qq = np.dot(q, q.T)
    bq = np.ones(dim)
    return np.linalg.solve(qq,bq)


def steady_state(
    p=np.matrix([
              [0,.2,.4,.4],
              [.2,0,.4,.4],
              [.2,.3,0,.5],
              [.3,.4,.3,0]
              ]),
    t=np.matrix([
          [1,2,3,1],
          [5,1,7,1],
          [1,1,1,1],
          [1,1,1,1]
          ])):
    '''
    IMPORTANT: p and t must be matrices and not 2d arrays.
    '''
    # How much average time does the system spend 
    # in each state once it gets to said state?
    p_times_t = np.array(np.sum(np.multiply(p,t),axis=1).T)[0]
    # Steady state probabilities based on p.
    pis = steady_state_props(p)
    # Need to weigh the steady state probs
    # based on p and re-normalize.
    nu_pis = (pis*p_times_t)/sum(pis*p_times_t)
    return nu_pis


if __name__ == "__main__":
  #PlotObjectiveFn(0.07044, 1.35254, 706) #Parameters taken from the output of FitDistributions.
	t = np.matrix(
	  [
	    [0, 0.012931074, 582.8588749, 528.527957,  30.17140938, 0, 213.8079142],
	    [0, 0, 148.3137143, 1.611737957, 17927.469, 0, 0],
	    [0, 0.301992986, 0, 934.5181613, 627.6994106, 0, 0],
	    [0, 151.4411604, 1222.75547,  0, 1016.439238, 0, 0],
	    [0, 0.021790501, 0, 0, 0, 0, 0],
	    [0, 0, 0.001, 0, 0.001, 0, 0],
	    [0, 14.73930389, 798.96504, 885.5214931, 770.4508371, 89616.33813, 0]
	  ]
	)
	p = np.matrix(
	  [
	    [0, 0.732362661, 0.944081099, 0.425732737, 0.99928845,  0.000613121, 1],
	    [0, 0, 0.000940157, 0.459016393, 1.42E-06,  0, 0],
	    [0, 0.000133541, 0, 0.014654744, 0.000181173, 0, 0],
	    [0, 2.80E-05,  0.027550687, 0, 0.000419186, 0, 0],
	    [0, 0.267462596, 0, 0, 0, 0, 0],
	    [0, 1.81E-07,  0.002820471, 0, 1.42E-06,  0, 0],
	    [0, 1.30E-05,  2.46E-02,  1.01E-01,  1.08E-04,  9.99E-01,  0]
	  ]
	)
	p[0,0] = 1
	t = np.transpose(t)
	p = np.transpose(p)
	def normalize(x):
	    return x/sum(x)
	p = np.apply_along_axis(normalize, axis = 1, arr = p)
	montecarlo(p, t, 3, 0)

