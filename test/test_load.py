import numpy as np
import GPy
import BBMM
import unittest
import os


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
bbmm_covariance_kernels = {'rbf': BBMM.kern.RBF, 'matern32': BBMM.kern.Matern32, 'matern52': BBMM.kern.Matern52}


class Test(unittest.TestCase):
    def _run(self, kernelname):
        bbmm_kernel = BBMM.kern.FullDerivative(bbmm_covariance_kernels[kernelname], 3, d)
        kern = GPy.kern.src.bbmm_kern.GPyKern(bbmm_kernel, variance=variance, lengthscale=lengthscale)
        model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=1e-4)
        model.optimize()
        model_dict = {
            'modelfile': model.to_dict()
        }
        np.savez("./saved_model", **model_dict)
        modelfile = dict(np.load("./saved_model.npz", allow_pickle=True))['modelfile'][()]
        model_load = GPy.core.model.Model.from_dict(modelfile)
        err = np.max(np.abs(model_load.predict(X)[0] - model.predict(X)[0]))
        os.remove("./saved_model.npz")
        self.assertTrue(err < 1e-8)

    def test_RBF(self):
        self._run('rbf')

    def test_Matern32(self):
        self._run('matern32')

    def test_Matern52(self):
        self._run('matern52')


if __name__ == '__main__':
    unittest.main()
