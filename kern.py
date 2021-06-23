import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cupy as cp


class Kernel(object):
    def __init__(self):
        pass

    def K(self, X1, X2=None):
        raise NotImplementedError

    def dK_dlengthscale(self, X1, X2=None):
        raise NotImplementedError


class Stationary(Kernel):
    def __init__(self):
        super().__init__()

    def set_lengthscale(self, lengthscale):
        self.lengthscale = lengthscale

    def set_variance(self, variance):
        self.variance = variance

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

    def K(self, X1, X2=None):
        r = self.r(X1, X2)
        return self.K_of_r(r)

    def K_of_r(self, r):
        raise NotImplementedError

    def dK_dr(self, r):
        raise NotImplementedError

    def dK_dlengthscale(self, X1, X2=None):
        r = self.r(X1, X2)
        return -self.dK_dr(r) * r / self.lengthscale


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

    def K_of_r(self, r):
        xp = cp.get_array_module(r)
        return xp.exp(-r**2 / 2) * self.variance

    def dK_dr(self, r):
        xp = cp.get_array_module(4)
        return -xp.exp(-r**2 / 2) * r * self.variance


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
