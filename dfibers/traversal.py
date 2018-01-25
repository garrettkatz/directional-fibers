import time
import numpy as np
import numerical_utilities as nu
import itertools as it
import matplotlib.pyplot as plt

def eF(x, c, f, ef):
    """
    Forward error in F(x)
    x: point on fiber
    c, f, ef as in traverse_fiber
    """
    v, a = x[:-1,:], x[-1]
    error = nu.eps(f(v) - a*c) + ef(v) + nu.eps(a*c) + nu.eps(a)*c
    return error

def refine_initial(f, Df, x, c, max_solve_iterations, solve_tolerance):
    residuals = []
    for i in it.count(1):
        v, a = x[:-1,:], x[-1]
        F = f(v) - a*c
        residuals.append(np.fabs(F).max())
        if not np.isfinite(F).all(): break
        if (np.fabs(F) < solve_tolerance).all(): break
        DF = np.concatenate((Df(v), -c), axis=1)
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

def take_step(f, Df, c, z, x, step_amount, max_solve_iterations, solve_tolerance):
    N = c.shape[0]
    x0 = x
    x = x + z*step_amount # fast first step
    delta_g = np.zeros((N+1,1))
    Dg = np.zeros((N+1,N+1))
    Dg[:N,[N]] = -c
    Dg[[N],:] = z.T
    residuals = []
    for iteration in it.count(1):
        v, a = x[:-1,:], x[-1]
        delta_g[:N,:] = -(f(v) - a*c)
        delta_g[N,:] = step_amount - z.T.dot(x - x0)
        residuals.append(np.fabs(delta_g).max())
        if iteration >= max_solve_iterations: break
        if (np.fabs(delta_g) < solve_tolerance).all(): break
        Dg[:N,:N] = Df(v)
        x = x + nu.solve(Dg, delta_g)
    return x, residuals

def traverse_fiber(
    f,
    Df,
    compute_step_amount,
    v=None,
    c=None,
    z=None,
    N=None,
    terminate=None,
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
    The user provides functions f(v), Df(v), where v is N x 1
    The user-provided function compute_step_amount(x, DF, z) should return:
        step_amount: signed step size at point x along fiber with derivative DF and tangent z
        step_data: output for any additional data that is saved for post-traversal analysis
    v is an approximate starting point for traveral (defaults to the origin).
    c is a direction vector (defaults to random).
    z is an approximate initial tangent direction (automatically computed by default).
    N is the dimensionality of the dynamical system (defaults to shape of v, c, or z).
    At least one of v, c, and N should be provided.
    
    If provided, the function terminate(x) should return True when x meets a custom termination criterion.
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
    "status": one of "Terminated", "Closed loop", "Max steps", "Timed out", "Diverged".
    "X": X[:,n] is the n^{th} point along the fiber
    "residuals": residuals[n] is the residual error after Newton's method at the n^{th} step
    "step_amounts": step_amounts[n] is the size used for the n^{th} step
    "step_datas": step_datas[n] is the step_data saved at the n^{th} step
    "c": c is the direction vector that was used
    "z": z is the initial tangent vector that was used
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
        x[N,:] = (f(v)[c != 0] / c[c != 0]).mean()

    # Drive initial va to fiber in case of residual error
    x, initial_residuals = refine_initial(f, Df, x, c, max_solve_iterations, solve_tolerance)
    
    # Initialize outputs
    status = "Traversing"
    X = [x]
    residuals = [initial_residuals[-1]]
    step_amounts = []
    step_datas = []
    
    if not np.isfinite(initial_residuals[-1]).all():
        return {
            "status": "Diverged",
            "X": x,
            "residuals": np.array([initial_residuals[-1]]),
            "step_amounts": np.array(step_amounts),
            "step_datas": step_datas,
            "c": c,
            "z": z,
        }
    
    # Initialize fiber tangent
    DF = np.concatenate((Df(x[:N,:]), -c), axis=1)
    z = compute_tangent(DF, z=z)
    z_init = z

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
        if terminate is not None and terminate(x):
            status = "Terminated"
            break            
        # Check for closed loop
        if len(X) > 2 and np.fabs(X[-1]-X[0]).max() < np.fabs(X[2]-X[0]).max():
            status = "Closed loop"
            break

        # Update DF
        DF = np.concatenate((Df(x[:N,:]), -c), axis=1)
        
        # Update tangent
        z = compute_tangent(DF, z)

        # Get step size
        step_amount, step_data = compute_step_amount(x, DF, z)
        if max_step_size is not None:
            step_amount = np.sign(step_amount)*min(np.fabs(step_amount), max_step_size)
       
        # Update x
        x, step_residuals = take_step(f, Df, c, z, x, step_amount, max_solve_iterations, solve_tolerance)

        # Store progress
        X.append(x)
        residuals.append(step_residuals[-1])
        step_amounts.append(step_amount)
        step_datas.append(step_data)
        
    # final output
    return {
        "status": status,
        "X": np.concatenate(X,axis=1),
        "residuals": np.array(residuals),
        "step_amounts": np.array(step_amounts),
        "step_datas": step_datas,
        "c": c,
        "z": z_init,
    }

def plot_fiber(X, Y, V, f, ax=None, scale_XY=1, scale_V=1):
    """
    Plots a fiber within a 2d state space
    pt.show still needs to be called separately
    X, Y: 2d ndarrays as returned by np.meshgrid
    V: (2,P) ndarray of P points along the fiber
    f: as in traverse_fiber
    ax: axis on which to draw
    """

    # Calculate direction vectors
    XY = np.array([X.flatten(), Y.flatten()])
    C_XY = f(XY)
    C_V = f(V)

    # Set up draw axes
    if ax is None: ax = plt.gca()

    # Draw ambient direction vectors
    ax.quiver(XY[0,:],XY[1,:],C_XY[0,:],C_XY[1,:],color=0.5*np.ones((1,3)),
        scale=scale_XY,units='xy',angles='xy')

    # Draw fiber with incident direction vectors
    ax.plot(V[0,:],V[1,:],'k-',linewidth=1)
    ax.quiver(V[0,:],V[1,:],C_V[0,:],C_V[1,:],color=0.0*np.ones((1,3)),
        scale=scale_V,units='xy',angles='xy')

    # Set limits based on grid and show
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    ax.set_xlim((X.min(),X.max()))
    ax.set_ylim((Y.min(),Y.max()))
