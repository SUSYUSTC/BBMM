import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cupy as cp


class Cache(object):
    def __init__(self):
        pass

    def __call__(self, f):
        def g(*args, **kwargs):
            self_f = args[0];
            name = f.__name__
            if not hasattr(self_f, 'cache'):
                self_f.cache = None
            if not hasattr(self_f, 'cache_data'):
                self_f.cache_data = {}
            if 'cache' in kwargs:
                do_cache = kwargs['cache']
            else:
                do_cache = self_f.cache

            self_f.cache = do_cache
            if do_cache:
                if name not in self_f.cache_data:
                    if 'cache' in kwargs:
                        del kwargs['cache']
                    self_f.cache_data[name] = f(*args, **kwargs)
                return self_f.cache_data[name]
            else:
                if 'cache' in kwargs:
                    del kwargs['cache']
                return f(*args, **kwargs)
        g.__name__ = f.__name__
        return g


class Kernel(object):
    def __init__(self):
        self.cache = False
        pass

    def K(self, X1, X2=None):
        raise NotImplementedError

    def dK_dl(self, X1, X2=None):
        raise NotImplementedError

    def dK_dv(self, X1, X2=None):
        raise NotImplementedError

    def d2K_dXdl(self, X1, dX1, X2=None):
        raise NotImplementedError

    def d2K_dXdv(self, X1, dX1, X2=None):
        raise NotImplementedError

    def d3K_dXdX2dl(self, X1, dX1, dX2, X2=None):
        raise NotImplementedError

    def d3K_dXdX2dv(self, X1, dX1, dX2, X2=None):
        raise NotImplementedError


class Stationary(Kernel):
    def __init__(self):
        super().__init__()

    def set_lengthscale(self, lengthscale):
        self.lengthscale = lengthscale

    def set_variance(self, variance):
        self.variance = variance

    def grad(self, X1, X2=None):
        return {
            'lengthscale': self.dK_dlengthscale(X1, X2),
            'variance': self.dK_dvariance(X1, X2)
        }

    def K_of_r(self, r):
        raise NotImplementedError

    def dK_dr(self, r):
        raise NotImplementedError

    def d2K_drdr(self, r):
        raise NotImplementedError

    def d3K_drdrdr(self, r):
        raise NotImplementedError

    @Cache()
    def r(self, X1, X2=None):
        '''
        A numpy-cupy generic code to calculate the distance matrix between X1 and X2.

        Parameters
        ----------
        X1: N*f array.
        X2: None or N*f array. X2=X1 if None is specified.
        '''
        xp = cp.get_array_module(X1)
        if X2 is None:
            X2 = X1
        N1 = len(X1)
        N2 = len(X2)
        # (X1-X2)^2 = X1^2 + X2^2 - 2*X1*X2
        distance = xp.empty((N1, N2))
        X11 = xp.square(X1).sum(1)
        X22 = xp.square(X2).sum(1)
        distance = X11[:, None] + X22[None, :]
        distance -= X1.dot(X2.T) * 2
        # Add a small number to avoid calculating square root of a negative number
        distance += 1e-12
        return xp.sqrt(distance) / self.lengthscale

    @Cache()
    def K(self, X1, X2=None):
        r = self.r(X1, X2)
        return self.K_of_r(r)

    #@Cache()
    def Xdiff_dX(self, X1, X2, dX1, r):
        xp = cp.get_array_module(X1)
        return xp.sum((X1[:, None, :] - X2[None, :, :]) * dX1[:, None, :], axis=-1)

    #@Cache()
    def Xdiff_dX2(self, X1, X2, dX2, r):
        xp = cp.get_array_module(X1)
        return xp.sum((X2[None, :, :] - X1[:, None, :]) * dX2[None, :, :], axis=-1)

    #@Cache()
    def dr_dX(self, X1, X2, dX1, r):
        if X2 is None:
            X2 = X1
        return self.Xdiff_dX(X1, X2, dX1, r) / r

    #@Cache()
    def dr_dX2(self, X1, X2, dX2, r):
        if X2 is None:
            X2 = X1
        return self.Xdiff_dX2(X1, X2, dX2, r) / r

    #@Cache()
    def d2r_dXdX2(self, X1, X2, dX1, dX2, r):
        if X2 is None:
            X2 = X1
        xp = cp.get_array_module(X1)
        return (-self.dr_dX(X1, X2, dX1, r) * self.dr_dX2(X1, X2, dX2, r) - dX1.dot(dX2.T)) / r
        #return 0

    @Cache()
    def dr_dl(self, r):
        return -r / self.lengthscale

    @Cache()
    def dK_dl(self, X1, X2=None):
        r = self.r(X1, X2)
        return self.dK_dr(r) * self.dr_dl(r)

    @Cache()
    def d2K_drdl(self, r):
        return self.d2K_drdr(r) * self.dr_dl(r) - self.dK_dr(r) / self.lengthscale

    @Cache()
    def d3K_drdrdl(self, r):
        return self.d3K_drdrdr(r) * self.dr_dl(r) - self.d2K_drdr(r) / self.lengthscale * 2

    @Cache()
    def dK_dv(self, X1, X2=None):
        return self.K(X1, X2) / self.variance

    @Cache()
    def d2K_drdv(self, r):
        return self.dK_dr(r) / self.variance

    @Cache()
    def d3K_drdrdv(self, r):
        return self.d2K_drdr(r) / self.variance

    # Start fake methods
    #@Cache()
    def _fake_dK_dX(self, method1, X1, dX1, X2=None):
        r = self.r(X1, X2)
        return method1(r) * self.dr_dX(X1, X2, dX1, r)

    #@Cache()
    def _fake_dK_dX2(self, method1, X1, dX2, X2=None):
        r = self.r(X1, X2)
        return method1(r) * self.dr_dX2(X1, X2, dX2, r)

    #@Cache()
    def _fake_d2K_dXdX2(self, method1, method2, X1, dX1, dX2, X2=None):
        r = self.r(X1, X2)
        return method2(r) * self.dr_dX(X1, X2, dX1, r) * self.dr_dX2(X1, X2, dX2, r) + method1(r) * self.d2r_dXdX2(X1, X2, dX1, dX2, r)

    # Start K
    #@Cache()
    def dK_dX(self, X1, dX1, X2=None):
        return self._fake_dK_dX(self.dK_dr, X1, dX1, X2=X2)

    #@Cache()
    def dK_dX2(self, X1, dX2, X2=None):
        return self._fake_dK_dX2(self.dK_dr, X1, dX2, X2=X2)

    #@Cache()
    def d2K_dXdX2(self, X1, dX1, dX2, X2=None):
        return self._fake_d2K_dXdX2(self.dK_dr, self.d2K_drdr, X1, dX1, dX2, X2=X2)

    # Start dK_dl
    #@Cache()
    def d2K_dXdl(self, X1, dX1, X2=None):
        return self._fake_dK_dX(self.d2K_drdl, X1, dX1, X2=X2)

    #@Cache()
    def d2K_dX2dl(self, X1, dX2, X2=None):
        return self._fake_dK_dX2(self.d2K_drdl, X1, dX2, X2=X2)

    #@Cache()
    def d3K_dXdX2dl(self, X1, dX1, dX2, X2=None):
        return self._fake_d2K_dXdX2(self.d2K_drdl, self.d3K_drdrdl, X1, dX1, dX2, X2=X2)

    # Start dK_dv
    #@Cache()
    def d2K_dXdv(self, X1, dX1, X2=None):
        return self._fake_dK_dX(self.d2K_drdv, X1, dX1, X2=X2)

    #@Cache()
    def d2K_dX2dv(self, X1, dX2, X2=None):
        return self._fake_dK_dX2(self.d2K_drdv, X1, dX2, X2=X2)

    #@Cache()
    def d3K_dXdX2dv(self, X1, dX1, dX2, X2=None):
        return self._fake_d2K_dXdX2(self.d2K_drdv, self.d3K_drdrdv, X1, dX1, dX2, X2=X2)


'''
    @Cache()
    def dK_dX(self, X1, dX1, X2=None):
        r = self.r(X1, X2)
        return self.dK_dr(r) * self.dr_dX(X1, X2, dX1, r)

    @Cache()
    def dK_dX2(self, X1, dX2, X2=None):
        r = self.r(X1, X2)
        return self.dK_dr(r) * self.dr_dX2(X1, X2, dX2, r)

    @Cache()
    def d2K_dXdX2(self, X1, dX1, dX2, X2=None):
        r = self.r(X1, X2)
        return self.d2K_drdr(r) * self.dr_dX(X1, X2, dX1, r) * self.dr_dX2(X1, X2, dX2, r) + self.dK_dr(r) * self.d2r_dXdX2(X1, X2, dX1, dX2, r)
'''


class RBF(Stationary):
    '''
    A RBF BBMM kernel.

    Parameters
    ----------
    X: N*f array.
    lengthscale: scalar.
    variance: scalar.
    noise: scalar.
    batch: Batch size of block kernel construction in order to save memory.
    nGPU: Number of used GPUs.
    file: A file to write the print information. Default to be standand console output.
    '''

    def __init__(self):
        super().__init__()

    @Cache()
    def K_of_r(self, r):
        xp = cp.get_array_module(r)
        return xp.exp(-r**2 / 2) * self.variance

    @Cache()
    def dK_dr(self, r):
        xp = cp.get_array_module(r)
        return -xp.exp(-r**2 / 2) * r * self.variance

    @Cache()
    def d2K_drdr(self, r):
        xp = cp.get_array_module(r)
        return xp.exp(-r**2 / 2) * (r**2 - 1) * self.variance

    @Cache()
    def d3K_drdrdr(self, r):
        xp = cp.get_array_module(r)
        return xp.exp(-r**2 / 2) * (3 - r**2) * r * self.variance


class Matern32(Stationary):
    '''
    A Matern32 BBMM kernel.

    Parameters
    ----------
    X: N*f array.
    lengthscale: scalar.
    variance: scalar.
    noise: scalar.
    batch: Batch size of block kernel construction in order to save memory.
    nGPU: Number of used GPUs.
    file: A file to write the print information. Default to be standand console output.
    '''

    def __init__(self):
        super().__init__()

    def K_of_r(self, r):
        xp = cp.get_array_module(r)
        return (1. + xp.sqrt(3.) * r) * xp.exp(-xp.sqrt(3.) * r) * self.variance

    def dK_dr(self, r):
        xp = cp.get_array_module(r)
        return - 3. * r * xp.exp(-xp.sqrt(3.) * r) * self.variance


class Matern52(Stationary):
    '''
    A Matern52 BBMM kernel.

    Parameters
    ----------
    X: N*f array.
    lengthscale: scalar.
    variance: scalar.
    noise: scalar.
    batch: Batch size of block kernel construction in order to save memory.
    nGPU: Number of used GPUs.
    file: A file to write the print information. Default to be standand console output.
    '''

    def __init__(self):
        super().__init__()

    def K_of_r(self, r):
        xp = cp.get_array_module(r)
        return (1 + xp.sqrt(5.) * r + 5. / 3 * r**2) * xp.exp(-xp.sqrt(5.) * r) * self.variance

    def dK_dr(self, r):
        xp = cp.get_array_module(r)
        return (- 5.0 / 3 * r - 5. * xp.sqrt(5.) / 3 * r**2) * xp.exp(-xp.sqrt(5.) * r) * self.variance
