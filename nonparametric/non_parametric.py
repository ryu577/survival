import numpy as np


def constr_matrices_data_distr(tau, ti, xi = None, intervention_cost = 199.997, distr = None):
    '''
    Uses the raw data to construct the transition probabilities and times matrices.
    Uses a parametric distribution only in instances where it has incomplete information.
    args:
        tau: The threshold at which the matrices are being constructed.
        ti: The observed samples.
        xi: The censored samples.
        intervention_cost: How much it costs to quit waiting and switch to plan B.
        distr: The distribution to be used when the sample is censored and the threshold
               being evaluated is greater than the censored value. In this case,
               we have no choice but to use a distribution since we can't know what
               exactly would have happened if it were not censored.
    '''
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
                tless = distr.expctd_x_bw_lts(x,tau) if pless > 1e-6 else 0
                t0 += np.array([0, tau, tless]) * np.array([0, pmore/(pmore+pless), pless/(pmore+pless)])
    t0[1] = t0[1] / p0[1] if p0[1] > 0 else 0
    t0[2] = t0[2] / p0[2] if p0[2] > 0 else 0
    p0 = p0 / sum(p0)
    probs = np.array([
      p0,
      [0,0,1],
      [0.0,1.0,0]
    ])
    times = np.array([
      t0,
      [0,0,intervention_cost],
      [5.0,5.0,0]
    ])
    return (probs, times)


def constr_matrices_dist(tau, intervention_cost, distr):
    '''
    Construct transition matrices from a distribution.
    args: 
        tau: The threshold.
        intervention_cost: The cost of switching to plan B and giving up on waiting.
        distr: the distribution based on which the transition matrices are to be calculated.        
    '''
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


def relative_nonparametric(recovery_times, current_censor = 600.0, intervention_cost = 200.0, verbose = False):
    '''
    Finds the optimal threshold given some data and making no assumption about the distribution.
    The approach is based on calculating the savings relative to the current threshold.
    Can only evaluate thresholds lower than the current threshold this way.
    args:
        recovery_times: The array of observed survival times.
        current_censor: The value at which the data is currently censored and relative to which the savings will be calculated.
        intervention_cost: The cost of giving up on waiting and switching to plan b.
        verbose: Print results for each 20th threshold evaluated or not.
    '''
    numOfReboots = sum(recovery_times > current_censor)
    relSavings = []
    taus = np.arange(10, current_censor, 5)
    indx = 0
    neglosses = []
    poslosses = []
    for tau in taus:
        indx += 1
        savings = numOfReboots * (current_censor - tau)
        losses = 0
        for i in recovery_times:
            if i > tau and i < current_censor:
                losses = losses + (tau + intervention_cost - i)
        netSavings = (savings - losses)
        relSavings.append(netSavings)
        if indx%20 == 0 and verbose:
            print("tau: " + "{0:.2f}".format(tau) + " savings: " + "{0:.2f}".format(savings) + " losses: " + "{0:.2f}".format(losses) + " net: " + "{0:.2f}".format(netSavings))
    return taus[np.argmax(relSavings)]



