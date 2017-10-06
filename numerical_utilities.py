import os
import numpy as np
import scipy.linalg as spl

def hardwrite(f,data):
    """
    Force file write to disk
    """
    if f.name == os.devnull: return
    f.write(data)
    f.flush()
    os.fsync(f)

def eps(x):
    """
    Returns the machine precision at x.
    I.e., the distance to the nearest distinct finite-precision number.
    Applies coordinate-wise if x is a numpy.array.
    """
    return np.fabs(np.spacing(x))

def mldivide(A, B):
    """
    Returns x, where x solves Ax = B. (A\B in MATLAB)
    """
    return np.linalg.lstsq(A,B)[0]

def mrdivide(B,A):
    """
    Returns x, where x solves B = xA. (B/A in MATLAB)
    """
    return np.linalg.lstsq(A.T, B.T)[0].T

def solve(A, B):
    """
    Returns x, where x solves Ax = B.
    Assumes A is invertible.
    """
    signature = 'dd->d'
    extobj = np.linalg.linalg.get_linalg_error_extobj(np.linalg.linalg._raise_linalgerror_singular)
    if B.ndim == A.ndim - 1:
        return np.linalg.linalg._umath_linalg.solve1(A, B, signature=signature, extobj=extobj)
    else:
        return np.linalg.linalg._umath_linalg.solve(A, B, signature=signature, extobj=extobj)

def minimum_singular_value(A):
    """
    Returns the minimum singular value of numpy.array A
    """
    # return np.linalg.norm(A, ord=-2)
    # return np.linalg.svd(A, compute_uv=0)[-1] # called deep within a code branch of np.linalg.norm
    return np.sqrt(spl.eigh(A.T.dot(A), eigvals_only=True, eigvals=(0,1))[0]) # slightly faster
