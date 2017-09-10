import os
import numpy as np

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

