class Kernel(object):
    def __init__(self):
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

    def K(self, X1, X2=None):
        raise NotImplementedError

    def clear_cache():
        raise NotImplementedError
