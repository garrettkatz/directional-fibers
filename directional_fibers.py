from utils import *
import itertools as it

def refine_initial(fDf, x, c, max_solve_iterations, solve_tolerance):
    for i in it.count(0):
        f, Df = fDf(x[:-1,:])
        F = f - x[-1]*c
        if i == max_solve_iterations: break
        if (np.fabs(F) < solve_tolerance).all(): break
        DF = np.concatenate((Df, -c), axis=1)
        x = x - mldivide(DF, F)
    return x, F, i

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
    x = refine_initial(fDf, x, c, max_solve_iterations, solve_tolerance)
    
    # Initialize fiber tangent
    f, Df = fDf(x[:N,:])
    DF = np.concatenate((Df, -c), axis=1)
    _,_,z = np.linalg.svd(DF)
    z = z[[N],:].T

    return x, z

    # Traverse
    X = []
    step_sizes = []
    s_mins = []
    residuals = []
    cloop_distance = np.nan
    # for step in it.count(0):

        # Get Df
        
        # Get z_new with solve (for Dg)

        # Get step size from Df and z_new (Dg)

        # Get va_new with take step (needs f and Df (g and Dg) on every iter)
        
        # Update va, z to va_new and z_new

        # Check local |alpha| minimum OR alpha sign change (neither implies the other in discretization)

        # Check for path termination criteria (asymptote in rnn)
            
        # Check for closed loop

        # Early termination criteria

        # final output
