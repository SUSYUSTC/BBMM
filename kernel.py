class Kernel(object):
    def __init__(self):
        self.cache = {}
        pass

    def K(self, X1, X2=None):
        raise NotImplementedError

    def dK_dl(self, X1, X2=None):
        raise NotImplementedError

    def dK_dv(self, X1, X2=None):
        raise NotImplementedError

    def d2K_dXdl(self, X1, dX1, X2=None):
        raise NotImplementedError

    def d2K_dXdv(self, X1, dX1, X2=None):
        raise NotImplementedError

    def d3K_dXdX2dl(self, X1, dX1, dX2, X2=None):
        raise NotImplementedError

    def d3K_dXdX2dv(self, X1, dX1, dX2, X2=None):
        raise NotImplementedError

    def clear_cache():
        raise NotImplementedError
