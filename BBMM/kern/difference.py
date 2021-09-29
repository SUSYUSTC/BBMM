from .kernel import Kernel
from .cache import Cache
from . import get_kern_obj


class Difference(Kernel):
    def __init__(self, kernel, d):
        self.name = 'difference.Difference'
        self.default_cache = {}
        self.d = d
        self.input_dim = d * 2
        self.dim_K = slice(0, d)
        self.dim_diff = slice(d, d*2)
        self.kernel = kernel
        self.nout = self.kernel.nout
        self.ps = self.kernel.ps
        self.set_ps = self.kernel.set_ps
        self.dK_dps = []
        for i in range(len(self.kernel.ps)):
            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp(i, X, X2, **kwargs)
            self.dK_dps.append(func)

        self.transformations = self.kernel.transformations
        super().__init__()
        self.kernel.set_cache_state(False)
        self.check()

    def likelihood_reg(self, X, noises):
        return super(self).likelihood_reg(X, noises)

    def likelihood_reg_grad(self, X, noises):
        return super(self).likelihood_reg_grad(X, noises)

    def _fake_K(self, X, X2, method):
        if X2 is None:
            X2 = X
        X_K = X[:, self.dim_K]
        X_diff = X[:, self.dim_diff]
        X2_K = X2[:, self.dim_K]
        X2_diff = X2[:, self.dim_diff]
        return method(X_K, X2_K) + method(X_K + X_diff, X2_K + X2_diff) - method(X_K, X2_K + X2_diff) - method(X_K + X_diff, X2_K)

    @Cache('g')
    def K(self, X, X2=None):
        return self._fake_K(X, X2, self.kernel.K)

    @Cache('g')
    def dK_dp(self, i, X, X2=None):
        return self._fake_K(X, X2, self.kernel.dK_dps[i])

    def clear_cache(self):
        self.cache_data = {}
        self.kernel.clear_cache()

    def set_cache_state(self, state):
        self.cache_state = state
        self.kernel.set_cache_state(state)

    def to_dict(self):
        data = {
            'name': self.name,
            'd': self.d,
            'kern': self.kernel.to_dict()
        }
        return data

    @classmethod
    def from_dict(self, data):
        d = data['d']
        kern_dict = data['kern']
        kernel = get_kern_obj(kern_dict)
        result = self(kernel, d)
        return result

