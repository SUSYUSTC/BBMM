class Kernel(object):
    def __init__(self):
        self.cache_state = True
        self.cache = {}
        pass

    def check(self):
        assert hasattr(self, 'default_cache')
        assert hasattr(self, 'ps')
        assert hasattr(self, 'set_ps')
        assert hasattr(self, 'dK_dps')
        assert hasattr(self, 'transform_ps')
        assert hasattr(self, 'd_transform_ps')
        assert hasattr(self, 'inv_transform_ps')

    def set_all_ps(self, params):
        assert len(params) == len(self.ps)
        for i in range(len(self.ps)):
            self.set_ps[i](params[i])

    def K(self, X1, X2=None):
        raise NotImplementedError

    def clear_cache():
        raise NotImplementedError
