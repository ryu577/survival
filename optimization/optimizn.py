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


def parabola_regrsn(a,b,c,x1,x2):
    """
    Parabola is given by ax^2+bx+c
    x1 and x2 are the two end points.
    """
    integ_fx = a*(x2**3-x1**3)/3+b*(x2**2-x1**2)/2+c*(x2-x1)
    integ_xfx = a*(x2**4-x1**4)/4+b*(x2**3-x1**3)/3+c*(x2**2-x1**2)/2
    a_l = (2*integ_xfx-(x2+x1)*integ_fx)/\
        (.6666667*(x2**3-x1**3)-1/2*(x2**2-x1**2)*(x2+x1))
    b_l = integ_fx/(x2-x1)-a_l*(x2+x1)/2
    return a_l, b_l


