# survival

All kinds of survival analysis distributions and methods to optimize how long to wait for them.


Say it takes ten minutes for you to walk to work. However, there is a bus that also takes you right from your house to work. As an added bonus, the bus has internet, so you can start working while on it. The catch is that you don’t know how long it will take for the bus to arrive.

Now, being the productive person you are, you want to minimize the time you spend being in a state where you can’t work (walking to work or waiting for the bus).


To install the library, you can run:

```
   pip install survival
```

Here is some sample code to help you get started.

```python
>>>from distributions.lomax import *
>>>from distributions.loglogistic import *
>>>k=1.1; lmb=0.5; sample_size=5000; censor_level=200; prob=1.0
# Generate the random data that needs to be fit.
>>>l = Lomax(k=k, lmb=lmb)
>>>samples = l.samples(size=sample_size)
>>>unifs = np.random.uniform(size=sample_size)
>>>ti = samples[(samples<=censor_level) + (unifs>prob)]
>>>xi = np.ones(sum( (samples>censor_level) * (unifs<=prob)))*censor_level
>>>ll1 = LogLogistic(ti=ti, xi=xi)
```

Most of the distributions covered here are also available in scipy.stats. So, why write a new library? 

There are many reasons, but here are some of them:
* The scipy classes don't have the ability to deal with censored data when fitting the distribution.
* The scipy classes don't have the ability to regress the distributions on features.
* The scipy classes are missing some methods crucial to the optimal waiting threshold problem, like hazard rate.



