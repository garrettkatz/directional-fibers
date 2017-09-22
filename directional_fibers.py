import numpy as np
import utils as ut
import itertools as it

def refine_initial(fDf, x, c, max_solve_iterations, solve_tolerance):
    residuals = []
    for i in it.count(1):
        f, Df = fDf(x[:-1,:])
        F = f - x[-1]*c
        residuals.append(np.fabs(F).max())
        if (np.fabs(F) < solve_tolerance).all(): break
        DF = np.concatenate((Df, -c), axis=1)
        x = x - ut.mldivide(DF, F)
        if i == max_solve_iterations: break
    return x, residuals

def update_tangent(DF, z):
    """
    Calculate the new tangent vector after the numerical step
    DF should be the Jacobian of F at the new point after the step (N by N+1 numpy.array)
    z should be the previous tangent vector before the step (N+1 by 1 numpy.array)
    returns z_new, the tangent vector after the step (N+1 by 1 numpy.array)
    """
    N = DF.shape[0]
    DG = np.concatenate((DF,z.T), axis=0)
    z_new = ut.solve(DG, np.concatenate((np.zeros((N,1)), [[1]]), axis=0)) # Fast DF null-space
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
    return sv_min / (2. * mu)

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
    If provided, each step uses at most max_solve_iterations of Newton's method.
    If provided, each step terminates after the Newton residual is within solve_tolerance.
    At least one of max_solve_iterations and solve_tolerance should be provided.

    A dictionary with the following entries is returned:
    "status": one of "Term", "Max steps", "Closed loop", "Timed out".
    "X": X[:,n] is the n^{th} point along the fiber
    "c": c is the direction vector that was used
    "step_sizes": step_sizes[n] is the size used for the n^{th} step
    "lambdas": lambdas[n] is the minimum singular value of Dg at the n^{th} step
    "residuals": residuals[n] is the residual error of Newton's method at the n^{th} step
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
        x[N,:] = a[a.isfinite()].mean()

    # Drive initial va to curve
    x, _ = refine_initial(fDf, x, c, max_solve_iterations, solve_tolerance)
    
    # Initialize fiber tangent
    f, Df = fDf(x[:N,:])
    DF = np.concatenate((Df, -c), axis=1)
    _,_,z = np.linalg.svd(DF)
    z = z[[N],:].T

    # Traverse
    X = []
    step_sizes = []
    s_mins = []
    residuals = []
    cloop_distance = np.nan
    for step in it.count(0):

        # Update DF
        f, Df = fDf(x[:N,:])
        DF = np.concatenate((Df, -c), axis=1)
        
        # Update tangent
        z_new = update_tangent(DF, z)

        # Get step size from Df and z_new (Dg)
        step_size = compute_step_size(mu, DF, z)

        # Get va_new with take step (needs f and Df (g and Dg) on every iter)
        
        # Update va, z to va_new and z_new

        # Check local |alpha| minimum OR alpha sign change (neither implies the other in discretization)

        # Check for path termination criteria (asymptote in rnn)
            
        # Check for closed loop

        # Early termination criteria

        # final output
