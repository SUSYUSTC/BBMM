import BBMM
import numpy as np
dtype = np.float64
X = np.load("./X_test.npy")
Y = np.load("./Y_test.npy")
N = len(Y)
lengthscale = 1.0
variance = 10.0
noise = 1e-4
batch = min(4096, N)
thres = 1e-4
# larger block size gives more accurate gradient
N_init = 500
kernel = BBMM.RBF()
bbmm = BBMM.BBMM(kernel, nGPU=1, file=None, verbose=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for bs in [10, 20, 50]:
    lengthscales = np.linspace(0.2, 0.8, num=50, endpoint=False)
    grads = []
    for lengthscale in lengthscales:
        bbmm.initialize(X, lengthscale, variance, noise, batch=batch)
        bbmm.set_preconditioner(N_init, nGPU=0)
        woodbury_vec_iter = bbmm.solve_iter(Y, thres=thres, block_size=bs, compute_gradient=True, random_seed=0, compute_loglikelihood=False, lanczos_n_iter=20, debug=False, max_iter=1000)
        #error = bbmm.predict(X, woodbury_vec_iter, training=True)-Y
        #print("MAX Error:", np.max(np.abs(error)))
        assert bbmm.converged
        grads.append(bbmm.gradients.lengthscale)
        print(lengthscale, end='\r')
    plt.plot(lengthscales, grads, label='bs='+str(bs))


plt.legend()
plt.xlabel('lengthscale')
plt.ylabel('gradient')
plt.savefig("grad_lengthscale.jpg")
