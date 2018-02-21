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

# def newton_raphson(f, Df,):
#     need finite check
#     need solve iterations
#     need to handle Df in stack form
#     non-square matrices, mldivide vs solve
#     switch to forward error instead of solve tolerance
#     return residuals


#     def refine_initial(f, Df, x, c, max_solve_iterations, solve_tolerance):
#         residuals = []
#         for i in it.count(1):
#             v, a = x[:-1,:], x[-1]
#             F = f(v) - a*c
#             residuals.append(np.fabs(F).max())
#             if not np.isfinite(F).all(): break
#             if (np.fabs(F) < solve_tolerance).all(): break
#             DF = np.concatenate((Df(v), -c), axis=1)
#             x = x - nu.mldivide(DF, F)
#             if i == max_solve_iterations: break
#         return x, residuals
    
#     def take_step(f, Df, c, z, x, step_amount, max_solve_iterations, solve_tolerance):
#         N = c.shape[0]
#         x0 = x
#         x = x + z*step_amount # fast first step
#         delta_g = np.zeros((N+1,1))
#         Dg = np.zeros((N+1,N+1))
#         Dg[:N,[N]] = -c
#         Dg[[N],:] = z.T
#         residuals = []
#         for iteration in it.count(1):
#             v, a = x[:-1,:], x[-1]
#             delta_g[:N,:] = -(f(v) - a*c)
#             delta_g[N,:] = step_amount - z.T.dot(x - x0)
#             residuals.append(np.fabs(delta_g).max())
#             if iteration >= max_solve_iterations: break
#             if (np.fabs(delta_g) < solve_tolerance).all(): break
#             Dg[:N,:N] = Df(v)
#             x = x + nu.solve(Dg, delta_g)
#         return x, residuals

#     def refine_points(V, f, ef, Df, max_iters=2**5, batch_size=100):
#         """
#         Refine candidate fixed points with Newton-Raphson iterations
#         V: (N,P)-ndarray, one candidate point per column
#         f, ef: as in is_fixed
#         Df(V)[p,:,:]: derivative of f(V[:,p])
#         max_iters: maximum Newton-Raphson iterations performed
#         batch_size: at most this many points solved for at a time (limits memory usage)
#         returns (V, fixed) where
#             V[:,p] is the p^th point after refinement
#             fixed[p] is True if the p^th point is fixed
#         """
    
#         # Split points into batches
#         num_splits = int(np.ceil(float(V.shape[1])/batch_size))
#         point_batches = np.array_split(V, num_splits, axis=1)
#         fixed_batches = []
    
#         for b in range(num_splits):
    
#             # Refine current batch with Newton-Raphson
#             points = point_batches[b]
#             fixed = np.zeros(points.shape[1], dtype=bool)
#             for i in range(max_iters):
#                 B = points[:,~fixed]
#                 DfB = Df(B)
#                 if len(DfB.shape)==2: DfB = DfB[np.newaxis,:,:]
#                 B = B - nu.solve(DfB, f(B).T).T
#                 points[:,~fixed] = B
#                 fixed[~fixed], _ = is_fixed(B, f, ef)
#                 if fixed.all(): break
#             point_batches[b] = points
#             fixed_batches.append(fixed)
    
#         # Format output
#         return np.concatenate(point_batches, axis=1), np.concatenate(fixed_batches)
