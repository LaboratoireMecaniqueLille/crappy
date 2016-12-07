import functools
import types
import warnings
import platform as _platform


def deprecated(replacement=None, warn_msg=""):
    """A decorator which can be used to mark functions as deprecated.
    replacement is a callable that replace the deprecated function.
    """
    warnings.simplefilter("once", PendingDeprecationWarning)

    def outer(fun):
        global msg
        if type(fun) == types.FunctionType:
            msg = " %s will be deprecated in the future. %s" % (
                '\033[93m' + fun.func_name + '\033[0m', '\033[1m' + warn_msg + '\033[0m')
        if replacement is not None:
            msg += "; use %s instead." % ('\033[93m' + replacement.func_name + '\033[0m')
        if fun.__doc__ is None:
            fun.__doc__ = msg

        @functools.wraps(fun)
        def inner(*args, **kwargs):
            warnings.warn(msg, PendingDeprecationWarning, stacklevel=3)
            return fun(*args, **kwargs)

        return inner

    return outer


warning_log = []


def import_error(warn_msg=""):
    warnings.simplefilter("once", ImportWarning)
    if _platform.system() == 'Linux':
        mess = '\033[93m' + warn_msg + '\033[0m'
    else:
        mess = warn_msg
    if warn_msg not in warning_log:
        warning_log.append(warn_msg)
        warnings.warn(mess, ImportWarning, stacklevel=2)
