import numpy as np

def bisection(fn, a=1e-6, b=1000):
    n = 1
    while n < 10000:
        c = (a + b) / 2
        if fn(c) == 0\
                    or (b - a) / 2 < 1e-6:
            return c
        n = n + 1
        if (fn(c) > 0) == \
            (fn(a) > 0):
            a = c
        else:
            b = c

