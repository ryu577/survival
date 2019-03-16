import numpy as np
from misc.sigmoid import *
from distributions.loglogistic import *
from distributions.regressed.basemodelregressed import *

class LogLogisticRegr(BaseRegressed):
    def __init__(self, ll):
        self.distr = ll
        self.shapeupper=5.0
        self.scaleupper=100.0


