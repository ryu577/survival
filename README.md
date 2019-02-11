# Survival

All kinds of survival analysis distributions and methods to optimize how long to wait for them.

Survival analysis is a branch of statistics for analyzing the expected duration of time until one or more events happen, such as death in biological organisms, failure in mechanical systems, occurrence of a disease, marriage, divorce, etc. 

## 1. Problem description

Let's take a real world example that you (actually I) face everyday. It takes ten minutes for you to walk to work. However, there is a bus that also takes you right from your house to work. As an added bonus, the bus has internet, so you can start working while on it. The catch is that you don’t know how long it will take for the bus to arrive.

Now, being the productive person you are, you want to minimize the time you spend being in a state where you can’t work (walking to work or waiting for the bus) over the long-run (say a year). How long should you wait for the bus each day given the distribution of its arrival times? 

There is a whole family of problems that can be expressed in this framework. Basically, this is for any scenario *where you're waiting for something*. For example, most of complex software components make API calls to other components. And, they have a plan B for when these calls fail. Now, how long should it wait for the API call to respond before treating the call as failure?

This library contains methods that can help:

1) Fit the distributions of the arrival times (for the bus or whatever process you're waiting for) given samples from it.
2) Handles censored data points. This basically means that sometimes, you only know the bus took more than (say) 10 minutes since you gave up waiting for it.
3) Find the optimal waiting thresholds using various strategies, parametric and non-parametric.
4) Optimizing multiple thresholds within a complex state machine to maximize steady state time spent in desirable states.


## 2. Installation
To install the library, run:

```
pip install survival
```

Make sure you have all the requirements (requirements.txt) installed. If not, you can run:

```
pip install -r requirements.txt
```

Alternately, you can fork/download the code and run from the main folder:

```
python setup.py install
```

In this case, you'll need to have a PYTHONPATH variable on your system, with the root folder of this project included in it.

## 3. Fitting a distribution to waiting times

In this section, we try and answer the question posed in section 1 - how long should we wait for the bus before giving up on it and starting to walk?

First, we'll need to observe some data on the historic arrival times of the bus and fit a distribution to them. Note however that some of our data will be incomplete since when we give up on the bus after x minutes, we only know it took more than that time for it to arrive, but not exactly how much. These are called censored observations.

Here is some sample code to fit a distribution when we have some complete observations (in array, ti) and some censored observations (in array xi).


```python
# If you don't have it already, you can install matplotlib via - 
# pip install matplotlib
>>> import matplotlib.pyplot as plt
>>> from distributions.lomax import *
>>> from distributions.loglogistic import *

# Parameters for Lomax
>>> k = 10.0; lmb = 0.5; sample_size = 5000; censor_level = 2.0; prob = 1.0

# For now, we just assume the arrival times of the bus follow a Lomax distribution.
>>> l = Lomax(k=k, lmb=lmb)

# Generate samples from Lomax distribution.
>>> samples = l.samples(size=sample_size)

# Since we never wait for the bus more than x minutes,
# the observed samples are the ones that take less than x minutes.
>>> ti = samples[(samples<=censor_level)]

# For the samples that took more than 10 minutes, add them to the censored array 
# all we know is they took more than x minutes but not exactly how long.
>>> xi = np.ones(sum(samples>censor_level))*censor_level

# Fit a log logistic model to the data we just generated (since we won't know the actual distribution in the real world, 
# we are fitting a distribution other than the one that generated the data). 
# Ignore the warnings.
>>> ll1 = LogLogistic(ti=ti, xi=xi)

# See how well the distribution fits the histogram.
>>> histo = plt.hist(samples, normed=True)
>>> xs = (histo[1][:len(histo[1])-1]+histo[1][1:])/2

>>> plt.plot(xs, [ll1.pdf(i) for i in xs])
>>> plt.show()
```

<a href="https://ryu577.github.io/jekyll/update/2018/05/22/optimum_waiting_thresholds.html" 
target="_blank"><img src="https://github.com/ryu577/ryu577.github.io/blob/master/Downloads/opt_thresholds/loglogistic_on_lomax.png" 
alt="Image formed by above method" width="240" height="180" border="10" /></a>

### 3.1 What is lomax distribution?


### 3.2 What is log logistics distribution?


## 4. Optimizing waiting threshold using the distribution

Going back the the waiting for a bus example, we can model the process as a state machine. There are three possible states that we care about - "waiting", "walking" and "working". The figure below represents the states and the arrows show the possible transitions between the states.

<a href="https://ryu577.github.io/jekyll/update/2018/05/22/optimum_waiting_thresholds.html" 
target="_blank"><img src="https://github.com/ryu577/ryu577.github.io/blob/master/Downloads/opt_thresholds/bus_states.png" 
alt="Image formed by above method" width="480" height="400" border="10" /></a>

Also, we assume that which state we go to next and how much time it takes to jump to that state depends only on which state we are currently in. This property is called the Markov property. 


To describe the transitions, we need two matrices. One for transition probabilities and another for transition times. The matrices represent some properties of the transition from state 'i' to state 'j' (exactly what will become clear soon). Again, the first state (i=0) is "waiting", the second state (i=1) is "walking" and the last and most desirable state where we want to spend the highest proportion of time is "working".


Continuing from above, we can run the following code:


```python
# The time it takes to walk to work 
intervention_cost=200

# The amount of time we wait for the bus before walking respectively.
tau=275

# The transition probabilities (p) and transition times (t) depend on 
# the amount of time we're willing to wait for the bus (tau)
# and the amount of time it takes to walk to work (intervention_cost).
>>> (p,t) = ll1.construct_matrices(tau, intervention_cost)

>>> p
matrix([[ 0.        ,  0.00652163,  0.99347837],
        [ 0.        ,  0.        ,  1.        ],
        [ 0.1       ,  0.9       ,  0.        ]])
```

The 'p' matrix you see above is the matrix of transition probabilities. The (i,j) row is the probability of transitioning to state 'j' given you started in state 'i'. Note that the rows of this matrix sum to 1 since we have to make a transition to one of the other available states. Also, the diagonals of the matrix are 0 since the (i,i) entry would imply transitioning from state i to i, which doesn't make sense in the context of a transition. 

Given this transition matrix, let's say we start in any state i and make a transition to another state according to the probabilities given by row 'i' in the matrix. If we end up in state 'j', we spend one unit of time there and then make another random transition according to the probabilities in row 'j' and so on, repeating this process many times. What percentage of the total time would we then expect to spend in each of the states? This is called the vector of steady state probabilities and it can be calculated via the method described in the answer <a href="https://math.stackexchange.com/a/2452452/155881">here</a>.


Now let's look at the other matrix we got. This is the matrix of transition times. The (i,j) entry represents that given we make a transition from state 'i' to state 'j', how long will it take on an average to make that transition. Again, the diagonals are zero since it doesn't make sense to transition from a state to itself. However, there is no constraint on the rows unlike the 'p' matrix.

```python
>>>t
matrix([[   0.        ,    5.        ,    0.27409621],
        [   0.        ,    0.        ,   20.        ],
        [ 100.        ,  100.        ,    0.        ]])
```

Now, instead of spending one unit of time in each state, what if we spend time amounting to the entry of (i,j) in the 't' matrix before making a transition to state 'j'? In the context of this process, what percentage of time do we expect to spend in each state? This is described <a href="https://math.stackexchange.com/a/2452464/155881">here</a>.

Notice how the code that generated the transition matrices is a function of the threshod, 'tau' and the intervention cost ('intervention_cost'). Now, we obviously want the 'tau' that gives us the most percentage of time spent in the state 'working'.

We can simply test many values of the threshold and pick the one that gives us the highest proportion of time spent in the 'working' state.

```python
>>> probs = []
>>> for tau in np.arange(10,900,1):
>>>	 (p,t) = ll1.construct_matrices(tau, intervention_cost)
>>>	 probs.append(steady_state(p, t)[2])
>>> opt_tau_1 = np.arange(10,900,1)[np.argmax(probs)]
```

Then, we can also calculate the optimal threshold <a href="https://github.com/ryu577/survival/blob/443e23d761656fad0069a3e0572d08f0706e8618/distributions/basemodel.py#L101">based on the parametric distribution</a>.


```python
opt_tau_2 = ll1.optimal_wait_threshold(intervention_cost)
```

And we can see that the two are very close to each other.

## 5. Why this library

Note: most of the distributions covered in this library are also available in scipy.stats. So, why write a new library? 

There are many reasons, but here are some of them:
* The scipy classes don't have the ability to deal with censored data when fitting the distribution.
* The scipy classes don't have the ability to regress the distributions on features.
* The scipy classes are missing some methods crucial to the optimal waiting threshold problem, like hazard rate.



