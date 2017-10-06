import time
import numpy as np
import numerical_utilities as nu
import itertools as it

def refine_initial(fDf, x, c, max_solve_iterations, solve_tolerance):
    residuals = []
    for i in it.count(1):
        f, Df = fDf(x[:-1,:])
        F = f - x[-1]*c
        residuals.append(np.fabs(F).max())
        if (np.fabs(F) < solve_tolerance).all(): break
        DF = np.concatenate((Df, -c), axis=1)
        x = x - nu.mldivide(DF, F)
        if i == max_solve_iterations: break
    return x, residuals

def compute_tangent(DF, z=None):
    """
    Compute the tangent vector to the directional fiber
    DF should be the Jacobian of F at the new point after the step (N by N+1 numpy.array)
    z should be None or the previous tangent vector before the step (N+1 by 1 numpy.array)
    returns z_new, the tangent vector after the step (N+1 by 1 numpy.array)
    if z is not None, the sign of z_new is selected for positive dot product with z
    """
    N = DF.shape[0]
    if z is None:
        _,_,z_new = np.linalg.svd(DF)
        z_new = z_new[[N],:].T
    else:
        DG = np.concatenate((DF,z.T), axis=0)
        z_new = nu.solve(DG, np.concatenate((np.zeros((N,1)), [[1]]), axis=0)) # Fast DF null-space
        z_new = z_new / np.sqrt((z_new**2).sum()) # faster than linalg.norm
    return z_new

def minimum_singular_value(A):
    """
    Returns the minimum singular value of numpy.array A
    """
    # return np.linalg.norm(A, ord=-2)
    return np.linalg.svd(A, compute_uv=0)[-1] # called deep within a code branch of np.linalg.norm

def compute_step_size(mu, DF, z):
    """
    Compute a step size at current traversal point
    mu should satisfy ||Df(x) - Df(y)|| <= mu * ||x - y||
    DF should be the Jacobian of F (an N by N+1 numpy.array)
    z should be the tangent vector (an N+1 by 1 numpy.array)
    """
    DG = np.concatenate((DF, z.T), axis=0)
    sv_min = minimum_singular_value(DG)
    return sv_min / (2. * mu), sv_min

def take_step(fDf, c, z, x, step_size, max_solve_iterations, solve_tolerance):
    N = c.shape[0]
    x0 = x
    x = x + z*step_size # fast first step
    delta_g = np.zeros((N+1,1))
    residuals = []
    for iteration in it.count(1):
        f, Df = fDf(x[:N,:])
        F = f - x[N,:]*c
        delta_g[:N,:] = -F
        delta_g[N,:] = step_size - z.T.dot(x - x0)
        residuals.append(np.fabs(delta_g).max())
        if iteration >= max_solve_iterations: break
        if (np.fabs(delta_g) < solve_tolerance).all(): break
        DF = np.concatenate((Df, -c), axis=1)
        Dg = np.concatenate((DF, z.T), axis=0)
        x = x + nu.solve(Dg, delta_g)
    return x, residuals

def traverse_fiber(
    fDf,
    mu,
    v=None,
    c=None,
    N=None,
    term=None,
    logfile=None,
    stop_time = None,
    max_traverse_steps=None,
    max_step_size=None,
    max_solve_iterations=None,
    solve_tolerance=None,
    ):

    """
    Traverses a directional fiber.
    All points/vectors represented as N x 1 or (N+1) x 1 numpy arrays
    The user-provided function fDf(v) should return f(v), Df(v).
    mu should satisfy ||Df(x) - Df(y)|| <= mu * ||x - y||
    v is an approximate starting point for traveral (defaults to the origin).
    c is a direction vector (defaults to random).
    N is the dimensionality of the dynamical system (defaults to shape of v or c).
    At least one of v, c, and N should be provided.
    
    If provided, the function term(x) should return True when x meets a custom termination criterion.
    If provided, progress is written to the file object logfile.
    If provided, traversal terminates at the clock time stop_time.
    If provided, traversal terminates after max_traverse_steps.
    If provided, step sizes are truncated to max_step_size.

    Each step is computed with Newton's method.
    Residual error is measured by the maximum norm of G.
    If provided, each step uses at most max_solve_iterations of Newton's method.
    If provided, each step terminates after the Newton residual is within solve_tolerance.
    At least one of max_solve_iterations and solve_tolerance should be provided.

    A dictionary with the following entries is returned:
    "status": one of "Terminated", "Closed loop", "Max steps", "Timed out".
    "X": X[n] is the n^{th} point along the fiber
    "residuals": residuals[n] is the residual error after Newton's method at the n^{th} step
    "step_sizes": step_sizes[n] is the size used for the n^{th} step
    "sv_mins": sv_mins[n] is the minimum singular value of Dg at the n^{th} step
    "c": c is the direction vector that was used
    """

    # Set defaults
    if v is not None: N = v.shape[0]
    if c is not None: N = c.shape[0]    
    if c is None:
        c = np.random.randn(N,1)
        c = c/np.sqrt((c**2).sum())    
    x = np.zeros((N+1,1))
    if v is not None: 
        x[:N,:] = v
        f, _ = fDf(x[:N,:])
        a = f/c
        x[N,:] = a[np.isfinite(a)].mean()

    # Drive initial va to fiber in case of residual error
    x, initial_residuals = refine_initial(fDf, x, c, max_solve_iterations, solve_tolerance)
    
    # Initialize fiber tangent
    f, Df = fDf(x[:N,:])
    DF = np.concatenate((Df, -c), axis=1)
    z = compute_tangent(DF)

    # Initialize outputs
    status = "Traversing"
    X = [x]
    residuals = [initial_residuals[-1]]
    step_sizes = []
    sv_mins = []

    # Traverse
    for step in it.count(0):

        # Check for early termination criteria
        if max_traverse_steps is not None and step >= max_traverse_steps:
            status = "Max steps"
            break
        if stop_time is not None and time.clock() >= stop_time:
            status = "Timed out"
            break
        # Check custom termination criteria
        if term is not None and term(x):
            status = "Terminated"
            break            
        # Check for closed loop
        if len(X) > 2 and np.fabs(X[-1]-X[0]).max() < np.fabs(X[2]-X[0]).max():
            status = "Closed loop"
            break

        # Update DF
        f, Df = fDf(x[:N,:])
        DF = np.concatenate((Df, -c), axis=1)
        
        # Update tangent
        z = compute_tangent(DF, z)

        # Get step size from Df and z_new (Dg)
        step_size, sv_min = compute_step_size(mu, DF, z)
        if max_step_size is not None: step_size = min(step_size, max_step_size)
       
        # Update x
        x, step_residuals = take_step(fDf, c, z, x, step_size, max_solve_iterations, solve_tolerance)

        # Store progress
        X.append(x)
        residuals.append(step_residuals[-1])
        step_sizes.append(step_size)
        sv_mins.append(sv_min)
        
    # final output
    return {
        "status":status,
        "X":X,
        "residuals":residuals,
        "step_sizes":step_sizes,
        "sv_mins":sv_mins,
        "c":c,
    }
