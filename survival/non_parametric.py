import numpy as np


def constr_matrices_data_distr(tau, ti, xi = None, intervention_cost = 199.997, distr = None):
    p0 = np.zeros(3)
    t0 = np.zeros(3)
    for i in ti:
        if i < tau:
            p0 += np.array([0, 0, 1.0])
            t0 += np.array([0, 0, i*1.0])
        else:
            p0 += np.array([0, 1.0, 0])
            t0 += np.array([0, tau*1.0, 0])
    if xi is not None and distr is not None:
        for x in xi:
            if tau < x:
                p0 += np.array([0, 1.0, 0])
                t0 += np.array([0, tau*1.0, 0])
            else:
                pless = distr.cdf(tau) - distr.cdf(x)
                pmore = distr.survival(tau)
                p0 += np.array([0, pmore/(pmore+pless), pless/(pmore+pless)])
                tless = distr.expectedXBwLts(x,tau) if pless > 1e-6 else 0
                t0 += np.array([0, tau, tless]) * np.array([0, pmore/(pmore+pless), pless/(pmore+pless)])
    t0[1] = t0[1] / p0[1] if p0[1] > 0 else 0
    t0[2] = t0[2] / p0[2] if p0[2] > 0 else 0
    p0 = p0 / sum(p0)
    probs = np.array([
      p0,
      [0,0,1],
      [0.5,0.5,0]
    ])
    times = np.array([
      t0,
      [0,0,intervention_cost],
      [5.0,5.0,0]
    ])
    return (probs, times)


def constr_matrices_dist(tau, intervention_cost, distr):
    x = 0
    pless = distr.cdf(tau) - distr.cdf(x)
    pmore = distr.survival(tau)
    tless = distr.expectedXBwLts(x,tau)
    t0 = np.array([0, tau, tless])
    p0 = np.array([0, pmore/(pmore+pless), pless/(pmore+pless)])
    probs = np.array(
    [
      p0,
      [0,0,1],
      [0.5,0.5,0]
    ])
    times = np.array(
    [
      t0,
      [0,0,intervention_cost],
      [5.0,5.0,0]
    ])
    return (probs, times)


def relative_nonparametric(recoveryTimes, currentCensor = 600.0, intervention_cost = 200.0, verbose = False):
    '''
    Finds the optimal threshold given some data and making no assumption about the distribution.
    '''
    numOfReboots = sum(recoveryTimes > currentCensor)
    relSavings = []
    taus = np.arange(10, currentCensor, 5)
    indx = 0
    neglosses = []
    poslosses = []
    for tau in taus:
        indx += 1
        savings = numOfReboots * (currentCensor - tau)
        losses = 0
        for i in recoveryTimes:
            if i > tau and i < currentCensor:
                losses = losses + (tau + intervention_cost - i)
        netSavings = (savings - losses)
        relSavings.append(netSavings)
        if indx%20 == 0 and verbose:
            print("tau: " + "{0:.2f}".format(tau) + " savings: " + "{0:.2f}".format(savings) + " losses: " + "{0:.2f}".format(losses) + " net: " + "{0:.2f}".format(netSavings))
    return taus[np.argmax(relSavings)]



