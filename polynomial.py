import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False
from .kernel import Kernel
from .cache import Cache


class Linear(Kernel):
    def __init__(self):
        super().__init__()
        self.name = 'polynomial.Linear'
        self.default_cache = {'g': 0}
        self.dK_dps = []
        self.ps = []
        self.set_ps = []
        self.transform_ps = []
        self.inv_transform_ps = []
        self.d_transform_ps = []
        self.check()

    def clear_cache(self):
        self.cache_data = {}

    @Cache('g')
    def K(self, X, X2=None):
        if X2 is None:
            return X.dot(X.T)
        else:
            return X.dot(X2.T)

    def to_dict(self):
        data = {
            'name': self.name
        }
        return data

    @classmethod
    def from_dict(self, data):
        kernel = self()
        return kernel

    def set_cache_state(self, state):
        self.cache_statue = state
