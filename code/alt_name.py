import functools
import inspect

def alt_name(other_name):
    def wrapper(f):
        @functools.wraps(f)
        def wraps_base_func(*args, **kwargs):
            return f(*args, **kwargs)
        ns = inspect.currentframe().f_back.f_locals
        ns[other_name] = f
        return wraps_base_func
    return wrapper