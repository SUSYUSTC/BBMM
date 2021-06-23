import BBMM
import numpy as np
import cupy as cp
import GPy
X = np.load("./X_test.npy")
Y = np.load("./Y_test.npy")
N = len(Y)
lengthscale = 1.0
variance = 10.0
noise = 1e-4
batch = min(4096, N)
thres = 1e-6
N_init = 500
bs = 400

BBMM_kernels = [BBMM.RBF(), BBMM.Matern32(), BBMM.Matern52()]
GPy_kernels = [GPy.kern.RBF, GPy.kern.Matern32, GPy.kern.Matern52]
names = ['RBF', 'Matern32', 'Matern52']
for i in range(3):
    print(names[i])
    bbmm_kernel = BBMM_kernels[i]
    bbmm = BBMM.BBMM(bbmm_kernel, nGPU=1, file=None, verbose=False)
    bbmm.initialize(X, lengthscale, variance, noise, batch=batch)
    bbmm.set_preconditioner(N_init, nGPU=1, debug=True)
    woodbury_vec_iter = bbmm.solve_iter(Y, thres=thres, block_size=bs, compute_gradient=True, random_seed=0, compute_loglikelihood=False, lanczos_n_iter=20, debug=False, max_iter=1000)

    gpy_kernel = GPy_kernels[i](input_dim=X.shape[1], lengthscale=lengthscale, variance=variance)
    gpy_model = GPy.models.GPRegression(X, Y, kernel=gpy_kernel, noise_var=noise)
    # 1e-6
    print("r(X):", np.max(np.abs(gpy_kernel._scaled_dist(X) - bbmm.kernel.r(X))))
    # 1e-10
    print("K(r):", np.max(np.abs(gpy_kernel.K_of_r(gpy_kernel._scaled_dist(X)) - bbmm.kernel.K_of_r(bbmm.kernel.r(X)))))

    pred = bbmm.predict(X, woodbury_vec_iter, training=True)
    # 1e-6
    print("prediction:", np.max(np.abs(pred - Y)))

    random_vectors = cp.asnumpy(bbmm.random_vectors)
    sampled_tr_I = np.sum(bbmm.Knoise_inv.dot(random_vectors) * random_vectors, axis=0)
    tr_I = np.mean(sampled_tr_I)
    # 1e-8
    print("tr(K_inv):", np.abs((bbmm.tr_I - tr_I) / tr_I))
    sampled_tr_dK_dlengthscale = np.sum(bbmm.Knoise_inv.dot(bbmm.dK_dlengthscale_full_np).dot(random_vectors) * random_vectors, axis=0)
    tr_dK_dlengthscale = np.mean(sampled_tr_dK_dlengthscale)
    # 1e-5
    print("tr(K_inv dK_dlengthscale):", np.abs((bbmm.tr_dK_dlengthscale - tr_dK_dlengthscale) / tr_dK_dlengthscale))

    # 1e-4
    print("variance:", np.abs((bbmm.gradients.variance - gpy_model.gradient[0]) / gpy_model.gradient[0]))
    # 1e-2
    print("lengthscale:", np.abs((bbmm.gradients.lengthscale - gpy_model.gradient[1]) / gpy_model.gradient[1]))
    # 1e-4
    print("noise:", np.abs((bbmm.gradients.noise - gpy_model.gradient[2]) / gpy_model.gradient[2]))
    print()
