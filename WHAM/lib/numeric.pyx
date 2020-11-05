"""Functions for dealing with correlated timeseries data"""
import numpy as np
import autograd.numpy as anp

cimport numpy as np


def alogsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Scipy logsumexp modified for autograd"""
    if b is not None:
        if anp.any(b == 0):
            a = a + 0.  # Promote to at least float
            a[b == 0] = -anp.inf

    a_max = anp.amax(a, axis=axis, keepdims=True)

    if b is not None:
        tmp = b * anp.exp(a - a_max)
    else:
        tmp = anp.exp(a - a_max)

    # Suppress warnings about log of zero
    with anp.errstate(divide='ignore'):
        s = anp.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = anp.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = anp.log(s)

    if not keepdims:
        a_max = anp.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def clogsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Cython-compiled Scipy logsumexp"""
    if b is not None:
        if np.any(b == 0):
            a = a + 0.  # Promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if b is not None:
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # Suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out
