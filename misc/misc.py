import numpy as np


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



