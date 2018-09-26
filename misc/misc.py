import numpy as np

def bisection(bisection_fn, a=1e-6, b=2000):
    n=1
    while n<10000:
        c=(a+b)/2
        if bisection_fn(c)==0 or (b-a)/2<1e-6:
            return c
        n = n + 1
        if (bisection_fn(c) > 0) == (bisection_fn(a) > 0):
            a=c
        else:
            b=c

def solve_hazard_eqn(fn, val, minval=10.0, maxval=900.0, interval=1.0):
	'''
	Finds the approximate point where a function crosses a value from below.
	'''
	prev_val = fn(minval)

	for i in np.arange(minval+interval, maxval, interval):
		next_val = fn(i)
		if next_val < val and val < prev_val:
			return i-0.5
		prev_val = next_val
	return maxval



