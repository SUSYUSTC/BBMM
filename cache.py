class Cache(object):
    def __init__(self, group='default'):
        self.group = group

    def __call__(self, f):
        def g(*args, **kwargs):
            self_f = args[0]
            name = f.__name__

            if not hasattr(self_f, 'cache'):
                self_f.cache = {}
            if not hasattr(self_f, 'cache_data'):
                self_f.cache_data = {}
            if name not in self_f.cache_data:
                self_f.cache_data[name] = {}
            this_func_cache_data = self_f.cache_data[name]

            if 'cache' in kwargs:
                this_cache = kwargs['cache']
            else:
                this_cache = self_f.cache
            self_f.cache = this_cache

            if self.group in this_cache:
                keyname = this_cache[self.group]
                if keyname not in this_func_cache_data:
                    if 'cache' in kwargs:
                        del kwargs['cache']
                    this_func_cache_data[keyname] = f(*args, **kwargs)
                else:
                    pass
                return this_func_cache_data[keyname]
            else:
                if 'cache' in kwargs:
                    del kwargs['cache']
                return f(*args, **kwargs)
        g.__name__ = f.__name__
        return g
