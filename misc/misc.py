import numpy as np


def solve_hazard_eqn(fn, val, minval=10.0, maxval=900.0, interval=1.0):
	'''
	Finds the approximate point where a function crosses a value from below.
	'''
	prev_val = fn(minval)

	for i in np.arange(minval+interval, maxval, interval):
		next_val = fn(i)
		if next_val < val and val < prev_val:
			return i-interval/2
		prev_val = next_val
	if next_val > val:
		return maxval
	else:
		return minval


def get_opt_tau(fn, pc_cost, q=1.0):
	pc_haz = q/pc_cost
	prev_haz = fn(9.5)
	max_haz = 0
	ans = 0.0
	for t in np.arange(10,900,0.5):
		curr_haz = fn(t)
		if curr_haz < pc_haz and prev_haz > pc_haz:
			return t-0.5/2
		prev_haz = curr_haz
		if curr_haz > max_haz:
			max_haz = curr_haz
	if max_haz < pc_haz:
		return 10
	else:
		return t


