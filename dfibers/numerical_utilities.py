import os
import itertools as it
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
    return np.linalg.lstsq(A,B,rcond=None)[0]

def mrdivide(B,A):
    """
    Returns x, where x solves B = xA. (B/A in MATLAB)
    """
    return np.linalg.lstsq(A.T, B.T,rcond=None)[0].T

def solve(A, B):
    """
    Returns x, where x solves Ax = B.
    Assumes A is invertible.
    If A is a KxNxN stack of matrices, B should be KxN.
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

def nr_solve(x, f, Df, ef, max_iterations=None):
    """
    Run Newton-Raphson iterations to solve f(x) = 0
    Inputs:
        x: an initial seed
        f: a function handle to the function f
        Df: a function handle computing the derivative of f
        ef: a function handle computing the forward error of f
        max_iterations: optional maximum number of iterations
    Returns:
        x: the final solution point
        points[i]: x at the i^th iteration
        residuals[i]: residual error at the i^th iteration
    """
    points = []
    residuals = []
    for i in it.count():
        fx = f(x)
        efx = ef(x)
        points.append(x)
        residuals.append(np.fabs(fx).max())
        if i == max_iterations or not np.isfinite(fx).all(): break
        if (np.fabs(fx) < efx).all(): break
        Dfx = Df(x)
        if fx.shape[0] < x.shape[0]:
            x = x - mldivide(Dfx, fx)
        if fx.shape[0] > x.shape[0]:
            x = x - mrdivide(Dfx, fx)
        if fx.shape[0] == x.shape[0]:
            x = x - solve(Dfx, fx)
    return x, points, residuals
        
def nr_solves(X, f, Df, ef, max_iterations=None):
    """
    Run Newton-Raphson iterations in parallel to solve multiple f(x) = 0
    Inputs:
        X[:,p]: the p^th initial seed
        f: a function handle to the function f
        Df: a function handle computing the derivative of f
            Df(X)[p,:,:] is the derivative of the p^th point
        ef: a function handle computing the forward error of f
        max_iterations: optional maximum number of iterations
    Returns:
        X: the final solution points
        done[i]: True iff the i^th point converged
        points[i]: X at the i^th iteration
        residuals[i]: maximum residual error at the i^th iteration
    """
    points = []
    residuals = []
    done = np.zeros(X.shape[1], dtype=bool)
    for i in it.count():
        fx = f(X[:,~done])
        efx = ef(X[:,~done])
        points.append(X)
        residuals.append(np.fabs(fx).max())
        if i == max_iterations or not np.isfinite(fx).all(): break
        done_now = (np.fabs(fx) < efx).all(axis=0)
        done[~done] = done_now
        if done_now.all(): break
        Dfx = Df(X[:,~done])
        fx = fx[:,~done_now]
        if fx.shape[1] > 1:
            X[:,~done] = X[:,~done] - solve(Dfx, fx.T).T
        else:
            X[:,~done] = X[:,~done] - solve(Dfx, fx)
    return X, done, points, residuals
