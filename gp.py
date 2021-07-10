import numpy as np
import cupy as cp
import scipy
import kern

X = np.load("../X_50.npy")[0:2000]
N = len(X)
Y = np.load("../Y_50.npy")[0:2000]
noise = 1e-4
k = kern.RBF()
k.set_lengthscale(1.0)
k.set_variance(1.0)
K = k.K(X)
K_noise = K + np.eye(N) * noise
K_noise_inv = np.linalg.inv(K_noise)
logdet = np.linalg.slogdet(K + np.eye(N) * noise)[1]
w = K_noise_inv.dot(Y)
ll = Y.T.dot(w)[0, 0] + logdet


class GP(object):
    def __init__(self, X, Y, kernel, noise=1e-4, GPU=False):
        self.kernel = kernel
        self.noise = noise
        self.GPU = GPU
        if GPU:
            self.xp = cp
        else:
            self.xp = np
        self.X = self.xp.array(X).copy()
        self.Y = self.xp.array(Y).copy()
        self.N = len(Y)

    def fit(self, grad=False):
        K = self.kernel.K(X)
        K_noise = K + self.xp.eye(self.N) * self.noise
        K_noise_inv = self.xp.linalg.inv(K_noise)
        w = K_noise_inv.dot(self.Y)
        logdet = self.xp.linalg.slogdet(K_noise)[1]
        ll = - (self.Y.dot(w)[0, 0] + logdet) / 2
