import numpy as np
try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False
from .kernel import Kernel
from .cache import Cache
from . import stationary


class FullDerivative(Kernel):
    def __init__(self, kernel, n, d):
        self.n = n
        self.d = d
        self.input_dim = (n + 1) * d
        self.dim_K = slice(0, self.d)
        self.dims_grad = [slice(self.d * (i + 1), self.d * (i + 2)) for i in range(self.n)]
        self.kern = kernel
        self.kernel = self.kern()
        self.lengthscale = 1.0
        self.variance = 1.0
        self.ps = [self.variance, self.lengthscale]
        self.set_ps = [self.set_variance, self.set_lengthscale]
        self.transform_ps = [self.transform_variance, self.transform_lengthscale]
        self.d_transform_ps = [self.d_transform_variance, self.d_transform_lengthscale]
        self.default_cache = {}
        self.name = 'adv_kern.FullDerivative'

    def set_variance(self, variance):
        self.kernel.set_variance(variance)
        self.variance = variance
        self.ps = [self.variance, self.lengthscale]

    def set_lengthscale(self, lengthscale):
        self.kernel.set_lengthscale(lengthscale)
        self.lengthscale = lengthscale
        self.ps = [self.variance, self.lengthscale]

    def transform_variance(self, variance):
        return np.log(variance)

    def transform_lengthscale(self, lengthscale):
        return np.log(lengthscale)

    def d_transform_variance(self, variance):
        return 1/variance

    def d_transform_lengthscale(self, lengthscale):
        return 1/lengthscale

    def inv_transform_variance(self, t_variance):
        return np.exp(t_variance)

    def inv_transform_lengthscale(self, t_lengthscale):
        return np.exp(t_lengthscale)

    def _fake_K(self, X, X2, K, dK_dX, dK_dX2, d2K_dXdX2):
        if gpu_available:
            xp = cp.get_array_module(X)
        else:
            xp = np
        if X2 is None:
            X2 = X
        N = len(X)
        N2 = len(X2)
        X_K = X[:, self.dim_K]
        X_grad = [X[:, dim] for dim in self.dims_grad]
        X2_K = X2[:, self.dim_K]
        X2_grad = [X2[:, dim] for dim in self.dims_grad]
        result = xp.zeros((N * (self.n + 1), N2 * (self.n + 1)))
        result[0:N, 0:N2] = K(X[:, self.dim_K], X2[:, self.dim_K], cache={'g': 0})
        for i in range(self.n):
            result[N * (i + 1): N * (i + 2), 0:N2] = dK_dX(X_K, X_grad[i], X2=X2_K, cache={'gd1': i, 'g': 0})
        for j in range(self.n):
            result[0:N, N2 * (j + 1):N2 * (j + 2)] = dK_dX2(X_K, X2_grad[j], X2=X2_K, cache={'gd2': j, 'g': 0})
        for i in range(self.n):
            for j in range(self.n):
                result[N * (i + 1): N * (i + 2), N2 * (j + 1):N2 * (j + 2)] = d2K_dXdX2(X_K, X_grad[i], X2_grad[j], X2=X2_K, cache={'gdd': (i, j), 'gd1': i, 'gd2': j, 'g': 0})
        return result

    @Cache('g')
    def K(self, X, X2=None):
        return self._fake_K(X, X2, self.kernel.K, self.kernel.dK_dX, self.kernel.dK_dX2, self.kernel.d2K_dXdX2)

    @Cache('g')
    def dK_dl(self, X, X2=None):
        return self._fake_K(X, X2, self.kernel.dK_dl, self.kernel.d2K_dXdl, self.kernel.d2K_dX2dl, self.kernel.d3K_dXdX2dl)

    @Cache('g')
    def dK_dv(self, X, X2=None):
        return self._fake_K(X, X2, self.kernel.dK_dv, self.kernel.d2K_dXdv, self.kernel.d2K_dX2dv, self.kernel.d3K_dXdX2dv)

    @Cache('g')
    def Kdiag(self, X):
        if gpu_available:
            xp = cp.get_array_module(X)
        else:
            xp = np
        X_K = X[:, self.dim_K]
        X_grad = [X[:, dim] for dim in self.dims_grad]
        return xp.concatenate([self.kernel.K_0(X_K)] + [self.kernel.d2K_dXdX_0(dX) for dX in X_grad])

    def dK_dldiag(self, X):
        if gpu_available:
            xp = cp.get_array_module(X)
        else:
            xp = np
        X_K = X[:, self.dim_K]
        X_grad = [X[:, dim] for dim in self.dims_grad]
        return xp.concatenate([self.kernel.dK_dl_0(X_K)] + [self.kernel.d3K_dldXdX_0(dX) for dX in X_grad])

    def clear_cache(self):
        self.cache_data = {}
        self.kernel.cache_data = {}

    def to_dict(self):
        data = {
            'name': self.name,
            'n': self.n,
            'd': self.d,
            'kern': self.kernel.to_dict()
        }
        return data

    @classmethod
    def from_dict(self, data):
        n = data['n']
        d = data['d']
        kern_dict = data['kern']
        kernel_type = eval(kern_dict['name'])
        result = FullDerivative(kernel_type, n, d)
        result.set_lengthscale(kern_dict['lengthscale'])
        result.set_variance(kern_dict['variance'])
        return result


class Derivative(Kernel):
    def __init__(self, kernel, n, d):
        self.n = n
        self.d = d
        self.input_dim = (n + 1) * d
        self.dim_K = slice(0, self.d)
        self.dims_grad = [slice(self.d * (i + 1), self.d * (i + 2)) for i in range(self.n)]
        self.kern = kernel
        self.kernel = self.kern()
        self.lengthscale = 1.0
        self.variance = 1.0
        self.ps = [self.variance, self.lengthscale]
        self.set_ps = [self.set_variance, self.set_lengthscale]
        self.transform_ps = [self.transform_variance, self.transform_lengthscale]
        self.d_transform_ps = [self.d_transform_variance, self.d_transform_lengthscale]
        self.default_cache = {}
        self.name = 'adv_kern.Derivative'

    def set_variance(self, variance):
        self.kernel.set_variance(variance)
        self.variance = variance
        self.ps = [self.variance, self.lengthscale]

    def set_lengthscale(self, lengthscale):
        self.kernel.set_lengthscale(lengthscale)
        self.lengthscale = lengthscale
        self.ps = [self.variance, self.lengthscale]

    def transform_variance(self, variance):
        return np.log(variance)

    def transform_lengthscale(self, lengthscale):
        return np.log(lengthscale)

    def d_transform_variance(self, variance):
        return 1/variance

    def d_transform_lengthscale(self, lengthscale):
        return 1/lengthscale

    def inv_transform_variance(self, t_variance):
        return np.exp(t_variance)

    def inv_transform_lengthscale(self, t_lengthscale):
        return np.exp(t_lengthscale)

    def _fake_K(self, X, X2, d2K_dXdX2):
        if gpu_available:
            xp = cp.get_array_module(X)
        else:
            xp = np
        if X2 is None:
            X2 = X
        N = len(X)
        N2 = len(X2)
        X_K = X[:, self.dim_K]
        X_grad = [X[:, dim] for dim in self.dims_grad]
        X2_K = X2[:, self.dim_K]
        X2_grad = [X2[:, dim] for dim in self.dims_grad]
        result = xp.zeros((N * self.n, N2 * self.n))
        for i in range(self.n):
            for j in range(self.n):
                result[N * i: N * (i + 1), N2 * j:N2 * (j + 1)] = d2K_dXdX2(X_K, X_grad[i], X2_grad[j], X2=X2_K, cache={'gdd': (i, j), 'gd1': i, 'gd2': j, 'g': 0})
        return result

    @Cache('g')
    def K(self, X, X2=None):
        return self._fake_K(X, X2, self.kernel.d2K_dXdX2)

    @Cache('g')
    def dK_dl(self, X, X2=None):
        return self._fake_K(X, X2, self.kernel.d3K_dXdX2dl)

    @Cache('g')
    def dK_dv(self, X, X2=None):
        return self._fake_K(X, X2, self.kernel.d3K_dXdX2dv)

    @Cache('g')
    def Kdiag(self, X):
        if gpu_available:
            xp = cp.get_array_module(X)
        else:
            xp = np
        X_grad = [X[:, dim] for dim in self.dims_grad]
        return xp.concatenate([self.kernel.d2K_dXdX_0(dX) for dX in X_grad])

    def dK_dldiag(self, X):
        if gpu_available:
            xp = cp.get_array_module(X)
        else:
            xp = np
        X_grad = [X[:, dim] for dim in self.dims_grad]
        return xp.concatenate([self.kernel.d3K_dldXdX_0(dX) for dX in X_grad])

    def clear_cache(self):
        self.cache_data = {}
        self.kernel.cache_data = {}

    def to_dict(self):
        data = {
            'name': self.name,
            'n': self.n,
            'd': self.d,
            'kern': self.kernel.to_dict()
        }
        return data

    @classmethod
    def from_dict(self, data):
        n = data['n']
        d = data['d']
        kern_dict = data['kern']
        kernel_type = eval(kern_dict['name'])
        result = Derivative(kernel_type, n, d)
        result.set_lengthscale(kern_dict['lengthscale'])
        result.set_variance(kern_dict['variance'])
        return result
