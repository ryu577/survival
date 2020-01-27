import numpy as np
from misc.sigmoid import *
from distributions.loglogistic import *
from distributions.regressed.basemodelregressed import *


class LogLogisticRegr(BaseRegressed):
    def __init__(self, t=None, x=None, fsamples=None, fcensored=None, ll=None):
        self.distr = ll
        self.shapeupper=5.0
        self.scaleupper=900.0
        if t is not None:
            super(LogLogisticRegr, self).set_state(t, x, fsamples, fcensored, ll)
        else:
            ti, xi, fsamples, fcensored = BaseRegressed.generate_data_(LogLogistic, 5000)
            ll = LogLogistic(ti, xi)
            super(LogLogisticRegr, self).set_state(ti, xi, fsamples, fcensored, ll)

