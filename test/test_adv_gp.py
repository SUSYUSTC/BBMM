import numpy as np
import GPy
import BBMM
import time
from GPy.core.parameterization import Param
import unittest


class NumericalDerivativeKernel(GPy.kern.Kern):
    def __init__(self, input_dim, n, kern, variance=1.0, lengthscale=1.0, active_dims=None, diff=1e-4):
        assert input_dim % (n + 1) == 0
        super(NumericalDerivativeKernel, self).__init__(input_dim, active_dims, 'derivative')
        self.n = n
        self.d = input_dim // (n + 1)
        self.dim_K = self.active_dims[0:self.d]
        self.dims_grad = [self.active_dims[self.d * (i + 1): self.d * (i + 2)] for i in range(self.n)]
        self.kern = kern
        self.kernel = kern(self.d, variance=variance, lengthscale=lengthscale)
        self.diff = diff
        self.diffX = diff * lengthscale
        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengthscale', lengthscale)
        self.link_parameters(self.variance, self.lengthscale)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        N = len(X)
        N2 = len(X2)
        X_K = X[:, self.dim_K]
        X_grad = [X[:, dim] for dim in self.dims_grad]
        X2_K = X2[:, self.dim_K]
        X2_grad = [X2[:, dim] for dim in self.dims_grad]
        result = np.zeros((N * (self.n + 1), N2 * (self.n + 1)))
        result[0:N, 0:N2] = self.kernel.K(X[:, self.dim_K], X2[:, self.dim_K])
        for i in range(self.n):
            dim_i = self.dims_grad[i]
            Kp = self.kernel.K(X_K + X_grad[i] * self.diffX, X2_K)
            Kn = self.kernel.K(X_K - X_grad[i] * self.diffX, X2_K)
            result[N * (i + 1): N * (i + 2), 0:N2] = (Kp - Kn) / (self.diffX * 2)
        for j in range(self.n):
            dim_j = self.dims_grad[j]
            Kp = self.kernel.K(X_K, X2_K + X2_grad[j] * self.diffX)
            Kn = self.kernel.K(X_K, X2_K - X2_grad[j] * self.diffX)
            result[0:N, N2 * (j + 1):N2 * (j + 2)] = (Kp - Kn) / (self.diffX * 2)
        for i in range(self.n):
            for j in range(self.n):
                dim_i = self.dims_grad[i]
                dim_j = self.dims_grad[j]
                Kpp = self.kernel.K(X_K + X_grad[i] * self.diffX, X2_K + X2_grad[j] * self.diffX)
                Kpn = self.kernel.K(X_K + X_grad[i] * self.diffX, X2_K - X2_grad[j] * self.diffX)
                Knp = self.kernel.K(X_K - X_grad[i] * self.diffX, X2_K + X2_grad[j] * self.diffX)
                Knn = self.kernel.K(X_K - X_grad[i] * self.diffX, X2_K - X2_grad[j] * self.diffX)
                result[N * (i + 1): N * (i + 2), N2 * (j + 1):N2 * (j + 2)] = (Kpp + Knn - Knp - Kpn) / (self.diffX * 2)**2
        return result

    def Kdiag(self, X):
        return np.diag(self.K(X, X))

    def dK_dl(self, X, X2=None):
        Kp = NumericalDerivativeKernel(self.input_dim, self.n, self.kern, variance=self.variance[0], lengthscale=self.lengthscale[0] * (1 + self.diff), active_dims=self.active_dims)
        Kn = NumericalDerivativeKernel(self.input_dim, self.n, self.kern, variance=self.variance[0], lengthscale=self.lengthscale[0] * (1 - self.diff), active_dims=self.active_dims)
        return (Kp.K(X, X2) - Kn.K(X, X2)) / (self.lengthscale * self.diff * 2)

    def dK_dv(self, X, X2=None):
        Kp = NumericalDerivativeKernel(self.input_dim, self.n, self.kern, variance=self.variance[0] * (1 + self.diff), lengthscale=self.lengthscale[0], active_dims=self.active_dims)
        Kn = NumericalDerivativeKernel(self.input_dim, self.n, self.kern, variance=self.variance[0] * (1 - self.diff), lengthscale=self.lengthscale[0], active_dims=self.active_dims)
        return (Kp.K(X, X2) - Kn.K(X, X2)) / (self.variance * self.diff * 2)

    def update_gradients_full(self, dL_dK, X, X2):
        dl = self.dK_dl(X, X2)
        dv = self.dK_dv(X, X2)
        self.variance.gradient = np.sum(dv * dL_dK)
        self.lengthscale.gradient = np.sum(dl * dL_dK)

    def parameters_changed(self):
        self.kernel = self.kern(self.d, variance=self.variance[0], lengthscale=self.lengthscale[0])
        pass


np.random.seed(0)
lengthscale = 1.0 + np.random.random()
variance = 1.0 + np.random.random()


n = 3
d = 2
N = 20


def y(X):
    Y = np.sum(np.sin(X[:, 0:d]), axis=1)
    Y_grad = [np.sum(X[:, d * (i + 1):d * (i + 2)] * np.cos(X[:, d * (i + 1):d * (i + 2)]), axis=1) for i in range(n)]
    return np.concatenate([Y] + Y_grad)[:, None]


np.random.seed(0)
X = np.random.random((N, (n + 1) * d))
X2 = np.random.random((10, 2))
Y = y(X)


class Test(unittest.TestCase):
    def _run(self, kernelname):
        kn = NumericalDerivativeKernel((n + 1) * d, n, getattr(GPy.kern, kernelname), lengthscale=lengthscale, variance=variance, diff=1e-4)
        ka = BBMM.kern.FullDerivative(getattr(BBMM.kern, kernelname), n, d)
        ka.set_lengthscale(lengthscale)
        ka.set_variance(variance)
        kda = BBMM.kern.Derivative(getattr(BBMM.kern, kernelname), n, d)
        kda.set_lengthscale(lengthscale)
        kda.set_variance(variance)
        err_K_diag = np.max(np.abs(np.diag(ka.K(X)) - ka.Kdiag(X)))
        err_dK_dl_diag = np.max(np.abs(np.diag(ka.dK_dl(X)) - ka.dK_dldiag(X)))

        K_a = ka.K(X)
        dK_dl_a = ka.dK_dl(X)
        dK_dv_a = ka.dK_dv(X)

        K_n = kn.K(X)
        dK_dl_n = kn.dK_dl(X)
        dK_dv_n = kn.dK_dv(X)

        K_da = kda.K(X)
        dK_dl_da = kda.dK_dl(X)
        dK_dv_da = kda.dK_dv(X)

        err_K = np.max(np.abs(K_a - K_n))
        err_dK_dl = np.max(np.abs(dK_dl_a - dK_dl_n))
        err_dK_dv = np.max(np.abs(dK_dv_a - dK_dv_n))
        self.assertTrue(np.max(np.abs(K_a[N:, N:] - K_da)) < 1e-8)
        self.assertTrue(np.max(np.abs(dK_dl_a[N:, N:] - dK_dl_da)) < 1e-8)
        self.assertTrue(np.max(np.abs(dK_dv_a[N:, N:] - dK_dv_da)) < 1e-8)
        if kernelname == 'Matern32':
            self.assertTrue(err_K < 2e-3)
            self.assertTrue(err_dK_dl < 2e-3)
            self.assertTrue(err_dK_dv < 2e-3)
            self.assertTrue(err_K_diag < 1e-5)
            self.assertTrue(err_dK_dl_diag < 1e-5)
        else:
            self.assertTrue(err_K < 1e-4)
            self.assertTrue(err_dK_dl < 1e-4)
            self.assertTrue(err_dK_dv < 1e-4)
            self.assertTrue(err_K_diag < 1e-10)
            self.assertTrue(err_dK_dl_diag < 1e-10)

    def test_RBF(self):
        self._run('RBF')

    def test_Matern32(self):
        self._run('Matern32')

    def test_Matern52(self):
        self._run('Matern52')


if __name__ == '__main__':
    unittest.main()
