import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import time
from .. import utils
if utils.gpu_available:
    import cupy as cp


class Preconditioner_Nystroem(object):
    '''
    A preconditioner of Nystroem kernel approximation
    '''

    def __init__(self, Lambda, U, diag_reg, nGPU=0):
        '''
        Parameters
        ----------
        Lambda: vector of size k. Eigenvalues of the rank-k preconditioner.
        U: N*k array. Corresponding eigenvectors.
        reg: scalar. The value that all zero eigenvalues are mapped to.
        nGPU: int. Number of used GPU. Using more than 1 GPUs is mainly used to reduce memory usage on each GPU but sometimes increase the computational time due to bandwidth.
        '''
        super().__init__()
        self.GPU = (nGPU > 0)
        self.nGPU = nGPU
        assert np.all(Lambda > 0), "Unexpected non-positive eigenvalues"
        if self.GPU:
            assert utils.gpu_available
            cp.cuda.Device(0).use()
        self.N, self.size, = U.shape
        # Timing
        self.total_time_U = 0.0
        self.total_time_UT = 0.0
        self.total_time_cpu2gpu = 0.0
        self.total_time_MMM = 0.0
        self.total_time_gpu2cpu = 0.0
        # Let M be the preconditioner, U and Lambda be its eigenvectors and eigenvalue matrix, R be the regulazation, then:
        # U Lambda UT ~ R^(-1/2) K R^(-1/2) 
        # K ~ R^(1/2) U Lambda UT R(1/2)
        # K_hat ~ M = R^(1/2) (U Lambda UT + I) R(1/2)
        # M^alpha = R^(alpha/2) [U ((Lambda+1)^(alpha) - 1) UT + I] R^(alpha/2)
        if self.GPU:
            # Divice all eigenvectors to several parts on different GPUs. If the number of GPU is larger than 1, only 1 eigenvector is stored on the GPU 0 and others are equally assigned to others.
            if self.nGPU == 1:
                self.device_division = [np.arange(self.N)]
            else:
                self.device_division = np.split(np.arange(self.N), [1] + [self.N * i // (self.nGPU - 1) for i in range(1, self.nGPU - 1)])
            self.Lambda = cp.asarray(Lambda)
            self.diag_reg = cp.asarray(diag_reg)
            self.U = []
            for i, d in enumerate(self.device_division):
                with cp.cuda.Device(i):
                    eigvec_on_device = cp.asarray(U[d])
                self.U.append(eigvec_on_device)
            cp.cuda.Stream.null.synchronize()
        else:
            self.Lambda = Lambda.copy()
            self.U = U.copy()
            self.diag_reg = diag_reg.copy()

    def dot(self, vec, transpose=False):
        '''
        A numpy-cupy generic code to perform self.U.dot(vec) or self.eigevecs.T.dot(vec)

        Parameters
        ----------
        vec: k*s array.
        transpose: bool.
        '''
        if not self.GPU:
            if transpose:
                return self.U.T.dot(vec)
            else:
                return self.U.dot(vec)
        else:
            assert vec.device.id == 0
            cp.cuda.Stream.null.synchronize()
            if transpose:
                batches = [None for i in range(self.nGPU)]
                vecs = [vec[self.device_division[i]] for i in range(self.nGPU)]
                cp.cuda.Stream.null.synchronize()
                t1 = time.time()
                # copy vec on GPU 0 to each device
                for i in range(self.nGPU):
                    with cp.cuda.Device(i):
                        vecs[i] = cp.asarray(vecs[i])
                cp.cuda.Stream.null.synchronize()
                t2 = time.time()
                # perform calculation
                for i in range(self.nGPU):
                    with cp.cuda.Device(i):
                        batches[i] = self.U[i].T.dot(vecs[i])
                cp.cuda.Stream.null.synchronize()
                t3 = time.time()
                # copy back to GPU 0 and sum up
                with cp.cuda.Device(0):
                    for i in range(self.nGPU):
                        batches[i] = cp.asarray(batches[i])
                result = cp.sum(cp.array(batches), axis=0)
                cp.cuda.Stream.null.synchronize()
                t4 = time.time()
                self.total_time_cpu2gpu += t2 - t1
                self.total_time_MMM += t3 - t2
                self.total_time_gpu2cpu += t4 - t3
                assert result.device.id == 0
                return result
            else:
                batches = [None for i in range(self.nGPU)]
                vecs = [None for i in range(self.nGPU)]
                cp.cuda.Stream.null.synchronize()
                t1 = time.time()
                # copy vec on GPU 0 to each device
                for i in range(self.nGPU):
                    with cp.cuda.Device(i):
                        vecs[i] = cp.asarray(vec)
                cp.cuda.Stream.null.synchronize()
                t2 = time.time()
                # perform calculation
                for i in range(self.nGPU):
                    with cp.cuda.Device(i):
                        batches[i] = self.U[i].dot(vecs[i])
                cp.cuda.Stream.null.synchronize()
                t3 = time.time()
                # copy back to GPU 0 and concatenate
                for i in range(self.nGPU):
                    with cp.cuda.Device(0):
                        batches[i] = cp.asarray(batches[i])
                result = cp.concatenate(batches, axis=0)
                cp.cuda.Stream.null.synchronize()
                t4 = time.time()
                self.total_time_cpu2gpu += t2 - t1
                self.total_time_MMM += t3 - t2
                self.total_time_gpu2cpu += t4 - t3
                assert result.device.id == 0
                return result

    def mv_alpha(self, v, alpha):
        '''
        # M^alpha = R^(alpha/2) [U ((Lambda+1)^(alpha) - 1) UT + I] R^(alpha/2)

        Parameters
        ----------
        v: k*s array.
        '''
        if v.ndim == 2:
            axis = (slice(None), None)
        else:
            axis = (slice(None), )
        xp = utils.get_array_module(v)
        alpha_Lambda = xp.power(self.Lambda + 1.0, alpha) - 1.0
        result = v.copy()
        result *= xp.power(self.diag_reg, alpha/2.0)[axis]
        result = self.dot(result, transpose=True)
        result *= alpha_Lambda[axis]
        result = self.dot(result, transpose=False)
        result *= xp.power(self.diag_reg, alpha/2.0)[axis]
        return result + v * xp.power(self.diag_reg, alpha)[axis]

    def mv_half(self, v):
        return self.mv_alpha(v, 0.5)

    def mv_invhalf(self, v):
        return self.mv_alpha(v, -0.5)
