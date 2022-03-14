import numpy as np
try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False
import numpy.linalg as LA
import time


class Krylov(object):
    def __init__(self, A, b, thres=1e-6, callback=None, lanczos_vectors=None, lanczos_n_iter=20, debug=False, max_iter=None, residual_check={}):
        self.A = A
        self.b = b
        self.thres = thres
        self.lanczos_n_iter = lanczos_n_iter
        if gpu_available:
            self.xp = cp.get_array_module(self.b)
        else:
            self.xp = np
        self.callback = callback
        self.debug = debug
        self.max_iter = max_iter
        self.b_norm = self.xp.linalg.norm(b, axis=0)
        self.n = len(self.b)
        self.x = self.xp.zeros_like(self.b)
        self.l = lanczos_vectors
        self.r = - self.b.copy()
        self.p = - self.r.copy()
        self.r_k_norm = self.r.T.dot(self.r)
        self.i = 0
        self.bcg_converged = False
        self.alphas = []
        self.betas = []
        if self.debug:
            self.ps = [self.p.copy()]
            self.rs = [self.r.copy()]
        if self.l is not None:
            assert self.xp.all(self.xp.isclose(self.xp.linalg.norm(self.l, axis=0), 1.0))
            self.d = []
            self.e = []
            self.previous = None
            self.lanczos_converged = False
        else:
            self.lanczos_converged = True
        self.residual_check = residual_check

    def compute_A(self):
        if (self.l is not None) and (not self.lanczos_converged):
            self.Ap, self.Al = self.A(self.p, self.l)
        else:
            self.Ap = self.A(self.p)
        self.i += 1

    def step_lanczos(self):
        self.lanczos_converged = self.i >= self.lanczos_n_iter
        self.d.append(self.xp.sum(self.Al * self.l, axis=0))
        self.newl = self.Al.copy()
        self.newl -= self.xp.sum(self.newl * self.l, axis=0)[None, :] * self.l
        if self.previous is not None:
            self.newl -= self.xp.sum(self.newl * self.previous, axis=0)[None, :] * self.previous
        self.newl /= LA.norm(self.newl, axis=0)[None, :]
        if not self.lanczos_converged:
            self.e.append(self.xp.sum(self.Al * self.newl, axis=0))
        self.previous = self.l.copy()
        self.l = self.newl.copy()

    def step_bcg(self):
        # self.Ap: N*k
        if gpu_available:
            cp.cuda.Stream.null.synchronize()
        t1 = time.time()
        self.denominator = self.xp.linalg.inv(self.p.T.dot(self.Ap))
        #self.oldalpha = self.xp.linalg.inv(self.p.T.dot(self.Ap)).dot(self.r_k_norm)
        self.alpha = self.denominator.dot(-self.p.T.dot(self.r))
        self.alphas.append(self.alpha)
        #assert self.xp.all(self.xp.isclose(self.alpha - self.oldalpha, 0, atol=1e-4))
        # alpha: k*krylov.k
        self.x += self.p.dot(self.alpha)
        self.r += self.Ap.dot(self.alpha)
        #assert(self.xp.all(self.xp.isclose(self.r.T.dot(self.p), 0, atol=1e-4)))
        #self.r_kplus1_norm = self.r.T.dot(self.r)
        #self.oldbeta = self.xp.linalg.inv(self.r_k_norm).dot(self.r_kplus1_norm)
        self.beta = self.denominator.dot(self.Ap.T.dot(self.r))
        self.betas.append(self.beta)
        #assert self.xp.all(self.xp.isclose(self.beta - self.oldbeta, 0, atol=1e-4))
        #self.r_k_norm = self.r_kplus1_norm
        self.residual = cp.max(self.xp.linalg.norm(self.r, axis=0) / self.b_norm)
        #self.residual = cp.asnumpy(self.r_kplus1_norm)[0, 0]
        self.p = self.p.dot(self.beta) - self.r
        #if self.i % 200 == 0:
        #    self.p = self.xp.random.random(self.p.shape)
        #self.p /= self.xp.linalg.norm(self.p, axis=0)[None, :]
        #self.p *= (self.xp.random.random(self.p.shape) * 1e-5 + 1)
        #assert(self.xp.all(self.xp.isclose(self.p.T.dot(self.Ap), 0, atol=1e-4)))
        if self.debug:
            self.ps.append(self.p.copy())
            self.rs.append(self.r.copy())
        if gpu_available:
            cp.cuda.Stream.null.synchronize()
        t2 = time.time()
        if self.callback is not None:
            self.callback(self.i, self.residual, t2 - t1)
        self.bcg_converged = (self.residual < self.thres)

    def run(self):
        while True:
            self.compute_A()
            if not self.bcg_converged:
                self.step_bcg()
            if not self.lanczos_converged:
                self.step_lanczos()
            if self.bcg_converged and self.lanczos_converged:
                break
            if self.i == 1:
                self.init_residual = self.residual
            if (self.i in self.residual_check) and (self.residual / self.init_residual > self.residual_check[self.i]):
                break
            if (self.max_iter is not None) and (self.i >= self.max_iter):
                break
        if self.l is not None:
            return self.x, self.xp.array(self.d), self.xp.array(self.e)
        else:
            return self.x
