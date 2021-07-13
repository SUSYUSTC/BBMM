import BBMM
import numpy as np
import GPy
import unittest

X = np.load("./X_test.npy")
Y = np.load("./Y_test.npy")
N = len(Y)
lengthscale = 1.0
variance = 10.0
noise = 1e-3
batch = min(4096, N)
thres = 1e-6
N_init = 500
bs = 100
BBMM_kernels = [BBMM.kern.RBF(), BBMM.kern.Matern32(), BBMM.kern.Matern52()]
GPy_kernels = [GPy.kern.RBF, GPy.kern.Matern32, GPy.kern.Matern52]


class Test(unittest.TestCase):
    def _run(self, i):
        bbmm_kernel = BBMM_kernels[i]
        bbmm = BBMM.BBMM(bbmm_kernel, nGPU=1, file=None, verbose=False)
        bbmm.initialize(X, lengthscale, variance, noise, batch=batch)
        bbmm.set_preconditioner(N_init, nGPU=1, debug=True)
        woodbury_vec_iter = bbmm.solve_iter(Y, thres=thres, block_size=bs, compute_gradient=True, random_seed=0, compute_loglikelihood=False, lanczos_n_iter=20, debug=False, max_iter=1000)

        gpy_kernel = GPy_kernels[i](input_dim=X.shape[1], lengthscale=lengthscale, variance=variance)
        gpy_model = GPy.models.GPRegression(X, Y, kernel=gpy_kernel, noise_var=noise)
        # 1e-6
        err_r = np.max(np.abs(gpy_kernel._scaled_dist(X) - bbmm.kernel.r(X)))
        self.assertTrue(err_r < 1e-5)
        # 1e-10
        err_K = np.max(np.abs(gpy_kernel.K_of_r(gpy_kernel._scaled_dist(X)) - bbmm.kernel.K_of_r(bbmm.kernel.r(X))))
        self.assertTrue(err_K < 1e-10)

        pred = bbmm.predict(X, woodbury_vec_iter, training=True)
        # 1e-6
        err_pred = np.max(np.abs(pred - Y))
        self.assertTrue(err_pred < 1e-6)

        try:
            random_vectors = bbmm.random_vectors.get()
        except BaseException:
            random_vectors = bbmm.random_vectors
        sampled_tr_I = np.sum(bbmm.Knoise_inv.dot(random_vectors) * random_vectors, axis=0)
        tr_I = np.mean(sampled_tr_I)
        # 1e-8
        err_tr_noise = np.abs((bbmm.tr_I - tr_I) / tr_I)
        self.assertTrue(err_tr_noise < 1e-8)
        sampled_tr_dK_dlengthscale = np.sum(bbmm.Knoise_inv.dot(bbmm.dK_dlengthscale_full_np).dot(random_vectors) * random_vectors, axis=0)
        tr_dK_dlengthscale = np.mean(sampled_tr_dK_dlengthscale)
        # 1e-5
        err_tr_lengthscale = np.abs((bbmm.tr_dK_dlengthscale - tr_dK_dlengthscale) / tr_dK_dlengthscale)
        self.assertTrue(err_tr_lengthscale < 1e-8)

        # 1e-4
        err_grad_variance = np.abs((bbmm.gradients.variance - gpy_model.gradient[0]) / gpy_model.gradient[0])
        self.assertTrue(err_grad_variance < 0.05)
        # 1e-2
        err_grad_lengthscale = np.abs((bbmm.gradients.lengthscale - gpy_model.gradient[1]) / gpy_model.gradient[1])
        self.assertTrue(err_grad_lengthscale < 0.05)
        # 1e-4
        err_grad_noise = np.abs((bbmm.gradients.noise - gpy_model.gradient[2]) / gpy_model.gradient[2])
        self.assertTrue(err_grad_noise < 0.05)

    def test_RBF(self):
        self._run(0)

    def test_Matern32(self):
        self._run(1)

    def test_Matern52(self):
        self._run(2)


if __name__ == '__main__':
    unittest.main()
