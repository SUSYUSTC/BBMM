import numpy as np
try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False
from GPy.core.parameterization import Param
from .kern import Kern


class GPyKern(Kern):
    def __init__(self, kernel, active_dims=None):
        super(GPyKern, self).__init__(kernel.input_dim, active_dims, 'BBMM')
        self.kernel = kernel
        self.variance = Param('variance', kernel.ps[0])
        self.lengthscale = Param('lengthscale', kernel.ps[1])
        self.link_parameters(self.variance, self.lengthscale)

    def K(self, X, X2=None):
        return self.kernel.K(X, X2)

    def dK_dv(self, X, X2=None):
        return self.kernel.dK_dps[0](X, X2)

    def dK_dl(self, X, X2=None):
        return self.kernel.dK_dps[1](X, X2)

    def Kdiag(self, X):
        return self.kernel.Kdiag(X)

    def dK_dldiag(self, X):
        return self.kernel.dK_dldiag(X)

    def update_gradients_full(self, dL_dK, X, X2):
        dl = self.dK_dl(X, X2)
        dv = self.dK_dv(X, X2)
        if gpu_available:
            xp = cp.get_array_module(X)
        else:
            xp = np
        self.variance.gradient = xp.sum(dv * dL_dK)
        self.lengthscale.gradient = xp.sum(dl * dL_dK)

    def parameters_changed(self):
        # nothing todo here
        self.kernel.set_ps[0](self.variance[0])
        self.kernel.set_ps[1](self.lengthscale[0])
        self.kernel.clear_cache()
        pass

    def to_dict(self):
        return {
            'class': 'GPy.kern.GPyKern',
            'name': 'gpykern',
            'kern': self.kernel.to_dict(),
            'active_dims': self.active_dims
        }

    def _build_from_input_dict(self, data):
        kern_dict = data['kern']
        import BBMM
        kernel = BBMM.kern.get_kern_obj(kern_dict)
        return GPyKern(kernel, active_dims=data['active_dims'])
