# survival

All kinds of survival analysis distributions and methods to optimize how long to wait for them.


Say it takes ten minutes for you to walk to work. However, there is a bus that also takes you right from your house to work. As an added bonus, the bus has internet, so you can start working while on it. The catch is that you don’t know how long it will take for the bus to arrive.

Now, being the productive person you are, you want to minimize the time you spend being in a state where you can’t work (walking to work or waiting for the bus)
over the long-run (say a year). How long should you wait for the bus each day given the distribution of its arrival times? 


There is a whole family of problems that can be expressed in this framework. Basically, any scenario where you're waiting for something. For example, most complex software components make API calls to other components. And, they probably have a plan B for when these calls fail. Now, how long should it wait for the API call to succeed before giving up on it?

This library contains methods that can help:

1) Fit the distributions of the arrival times given samples from it.
2) Handles censored data points. This basically means that sometimes, you only know the bus took more than (say) 10 minutes since you gave up waiting for it.
3) Find the optimal waiting thresholds using various strategies, parametric and non-parametric.
4) Optimizing multiple thresholds within a complex state machine to maximize steady state time spent in desirable states.

To install the library, run:

```
   pip install survival
```

Here is some sample code to help you get started.


```python
>>>from distributions.lomax import *
>>>from distributions.loglogistic import *
>>>k=1.1; lmb=0.5; sample_size=5000; censor_level=200; prob=1.0
# Initialize a Lomax distribution
>>>l = Lomax(k=k, lmb=lmb)
# Generate some samples from said Lomax distribution.
>>>samples = l.samples(size=sample_size)
# Since we never wait for the bus more than 10 minutes, the observed samples are the ones that take less than 10 minutes.
>>>ti = samples[(samples<=censor_level)]
# For the samples that took more than 10 minutes, add them to the censored array and all we know is they took more than 10 minutes but 
# not exactly how long.
>>>xi = np.ones(sum(samples>censor_level))*censor_level
# Fit a log logistic model to the data we just generated.
>>>ll1 = LogLogistic(ti=ti, xi=xi)
```

Most of the distributions covered here are also available in scipy.stats. So, why write a new library? 

There are many reasons, but here are some of them:
* The scipy classes don't have the ability to deal with censored data when fitting the distribution.
* The scipy classes don't have the ability to regress the distributions on features.
* The scipy classes are missing some methods crucial to the optimal waiting threshold problem, like hazard rate.



