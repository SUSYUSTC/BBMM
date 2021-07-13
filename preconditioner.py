import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False
    xp = np
import time


class Preconditioner_Nystroem(object):
    '''
    A preconditioner of Nystroem kernel approximation
    '''

    def __init__(self, eigvals, eigvecs, reg, nGPU=0):
        '''
        Parameters
        ----------
        eigvals: vector of size k. Eigenvalues of the rank-k preconditioner.
        eigvecs: N*k array. Corresponding eigenvectors.
        reg: scalar. The value that all zero eigenvalues are mapped to.
        nGPU: int. Number of used GPU. Using more than 1 GPUs is mainly used to reduce memory usage on each GPU but sometimes increase the computational time due to bandwidth.
        '''
        super().__init__()
        self.GPU = (nGPU > 0)
        self.nGPU = nGPU
        if self.GPU:
            cp.cuda.Device(0).use()
        self.N, self.size, = eigvecs.shape
        self.reg = reg
        # Timing
        self.total_time_eigvecs = 0.0
        self.total_time_eigvecsT = 0.0
        self.total_time_eigvals = 0.0
        self.total_time_cpu2gpu = 0.0
        self.total_time_MMM = 0.0
        self.total_time_gpu2cpu = 0.0
        # Let M be the preconditioner, U and \Lambda be its eigenvectors and eigenvalue matrix, then:
        # M^{1/2} v = U (\Lambda^{1/2}-reg^{1/2}) U^T v + reg^{1/2} * v
        # M^{-1/2} v = U (\Lambda^{-1/2}-reg^{-1/2}) U^T v + reg^{-1/2} * v
        if self.GPU:
            # Divice all eigenvectors to several parts on different GPUs. If the number of GPU is larger than 1, only 1 eigenvector is stored on the GPU 0 and others are equally assigned to others.
            if self.nGPU == 1:
                self.device_division = [np.arange(self.N)]
            else:
                self.device_division = np.split(np.arange(self.N), [1] + [self.N * i // (self.nGPU - 1) for i in range(1, self.nGPU - 1)])
            self.eigvals = cp.asarray(eigvals)
            self.half_eigvals = (cp.sqrt(self.eigvals) - cp.sqrt(self.reg))
            self.invhalf_eigvals = (cp.sqrt(1 / self.eigvals) - cp.sqrt(1 / self.reg))
            self.eigvecs = []
            for i, d in enumerate(self.device_division):
                with cp.cuda.Device(i):
                    eigvec_on_device = cp.asarray(eigvecs[d])
                self.eigvecs.append(eigvec_on_device)
            cp.cuda.Stream.null.synchronize()
        else:
            self.eigvals = eigvals.copy()
            self.half_eigvals = (np.sqrt(self.eigvals) - np.sqrt(self.reg))
            self.invhalf_eigvals = (np.sqrt(1 / self.eigvals) - np.sqrt(1 / self.reg))
            self.eigvecs = eigvecs.copy()

    def dot(self, vec, transpose=False):
        '''
        A numpy-cupy generic code to perform self.eigvecs.dot(vec) or self.eigevecs.T.dot(vec)

        Parameters
        ----------
        vec: k*s array.
        transpose: bool.
        '''
        if not self.GPU:
            if transpose:
                return self.eigvecs.T.dot(vec)
            else:
                return self.eigvecs.dot(vec)
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
                        batches[i] = self.eigvecs[i].T.dot(vecs[i])
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
                        batches[i] = self.eigvecs[i].dot(vecs[i])
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

    def mv_invhalf(self, v):
        '''
        Perform M^{-1/2} v = U (\Lambda^{-1/2}-reg^{-1/2}) U^T v + reg^{-1/2} * v

        Parameters
        ----------
        v: k*s array.
        '''
        if gpu_available:
            xp = cp.get_array_module(v)
        result = v
        t1 = time.time()
        result = self.dot(result, True)
        t2 = time.time()
        if v.ndim == 2:
            result *= self.invhalf_eigvals[:, None]
        else:
            result *= self.invhalf_eigvals
        t3 = time.time()
        result = self.dot(result)
        t4 = time.time()
        self.total_time_eigvecsT += t2 - t1
        self.total_time_eigvals += t3 - t2
        self.total_time_eigvecs += t4 - t3
        return result + v * xp.sqrt(1 / self.reg)

    def mv_half(self, v):
        '''
        Perform M^{1/2} v = U (\Lambda^{1/2}-reg^{1/2}) U^T v + reg^{1/2} * v

        Parameters
        ----------
        v: k*s array.
        '''
        if gpu_available:
            xp = cp.get_array_module(v)
        result = v
        t1 = time.time()
        result = self.dot(result, True)
        t2 = time.time()
        if v.ndim == 2:
            result *= self.half_eigvals[:, None]
        else:
            result *= self.half_eigvals
        t3 = time.time()
        result = self.dot(result)
        t4 = time.time()
        self.total_time_eigvecsT += t2 - t1
        self.total_time_eigvals += t3 - t2
        self.total_time_eigvecs += t4 - t3
        return result + v * xp.sqrt(self.reg)
