import time
import numpy as np
import numerical_utilities as nu
import itertools as it
import matplotlib.pyplot as plt
import fixed_points as fx

class FiberTrace:
    """
    A record of fiber traversal.  Has fields:
    status: "Terminated" | "Closed loop" | "Max steps" | "Timed out" | "Diverged"
    c: direction vector as N x 1 np.array
    x: current fiber point as (N+1) x 1 np.array
    DF: current fiber Jacobian as N x (N+1) np.array
    z: current fiber tangent vector as (N+1) x 1 np.array
    z_initial: initial fiber tangent vector as (N+1) x 1 np.array
    points[p]: the p^th fiber point as (N+1) x 1 np.array
    residuals[p]: the p^th residual error as float
    step_amounts[p]: the p^th step amount as float
    step_data[p]: additional data for the p^th step (user defined)
    candidates[p]: True if p^{th} point is candidate root
    """
    def __init__(self, c):
        self.status = "Traversing"
        self.c = c
        self.x = None
        self.DF = None
        self.z = None
        self.z_initial = None
        self.points = []
        self.tangents = []
        self.residuals = []
        self.step_amounts = []
        self.step_data = []
        self.candidates = np.empty(0, dtype=bool)
        self.sign_changes = np.empty(0, dtype=bool)
        self.alpha_mins = np.empty(0, dtype=bool)

    def index_candidates(self, abs_alpha_min = True):
        X = np.concatenate(self.points, axis=1)
        X = X[:,len(self.candidates):]
        fixed_index, sign_changes, alpha_mins = fx.index_candidates(
            X, abs_alpha_min)
        self.candidates = np.concatenate((self.candidates, fixed_index))
        self.sign_changes = np.concatenate((self.sign_changes, sign_changes))
        self.alpha_mins = np.concatenate((self.alpha_mins, alpha_mins))

    def halve_points(self, abs_alpha_min = True):

        # Update candidate index
        self.index_candidates(abs_alpha_min)

        # Keep all candidates and half non-candidates
        keep = self.candidates.copy()
        non_candidates = np.flatnonzero(self.candidates == False)
        keep_non_candidates = non_candidates[::2]
        keep[keep_non_candidates] = True
        
        # Set up pruning
        def prune(l):
            return [l[k] for k in range(len(keep)) if keep[k]]

        # Do pruning
        self.points = prune(self.points)
        self.tangents = prune(self.tangents)
        self.residuals = prune(self.residuals)
        self.step_amounts = prune(self.step_amounts)
        self.step_data = prune(self.step_data)
        self.candidates = self.candidates[keep]
        self.sign_changes = self.sign_changes[keep]
        self.alpha_mins = self.alpha_mins[keep]

def eF(x, c, f, ef):
    """
    Forward error in F(x)
    x: point on fiber
    c, f, ef as in traverse_fiber
    """
    v, a = x[:-1,:], x[-1]
    error = nu.eps(f(v) - a*c) + ef(v) + nu.eps(a*c) + nu.eps(a)*c
    return error

def refine_initial(f, Df, ef, x, c, max_solve_iterations):
    x, _, residuals = nu.nr_solve(x,
        f = lambda x: f(x[:-1]) - x[-1]*c,
        Df = lambda x: np.concatenate((Df(x[:-1])[0], -c), axis=1),
        ef = lambda x: eF(x, c, f, ef),
        max_iterations=max_solve_iterations)
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

def take_step(f, Df, ef, c, z, x, step_amount, max_solve_iterations):
    x0 = x
    x, _, residuals = nu.nr_solve(
        x0 + z*step_amount, # fast first step
        f = lambda x: np.concatenate((
            f(x[:-1]) - x[-1]*c,
            z.T.dot(x - x0) - step_amount), axis=0),
        Df = lambda x: np.concatenate((
            np.concatenate((Df(x[:-1])[0], -c), axis=1),
            z.T), axis=0),
        ef = lambda x: np.concatenate((
            eF(x, c, f, ef),
            nu.eps(z.T.dot(x)) + z.T.dot(nu.eps(x)) + nu.eps(z*x).sum()),
            axis = 0),
        max_iterations=max_solve_iterations)
    return x, residuals

def traverse_fiber(
    f,
    Df,
    ef,
    compute_step_amount,
    v=None,
    c=None,
    z=None,
    N=None,
    terminate=None,
    logger=None,
    stop_time = None,
    max_traverse_steps=None,
    max_step_size=None,
    max_solve_iterations=None,
    ):

    """
    Traverses a directional fiber.
    Traversal state is maintained in a FiberTrace object
    The user provides functions f(v), Df(v), ef(v) where v is an N x 1 np.array:
        f is the function, Df is its derivative, and ef is its forward error.
    The user-provided function compute_step_amount(trace) should return:
        step_amount: signed step size at point x along fiber with derivative DF and tangent z
        step_data: output for any additional data that will be saved for post-traversal analysis

    v is an approximate starting point for traveral (defaults to the origin).
    c is a direction vector (defaults to random).
    z is an approximate initial tangent direction (automatically computed by default).
    N is the dimensionality of the dynamical system (defaults to shape of v, c, or z).
    At least one of v, c, and N should be provided.
    
    If provided, the function terminate(trace) should return True when trace meets a custom termination criterion.
    If provided, progress is written to the Logger object logger.
    If provided, traversal terminates at the clock time stop_time.
    If provided, traversal terminates after max_traverse_steps.
    If provided, step sizes are truncated to max_step_size.

    Each step is computed with Newton's method.
    Residual error is measured by the maximum norm of G.
    If provided, each step uses at most max_solve_iterations of Newton's method.

    Returns the FiberTrace object for the traversal
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
    x, initial_residuals = refine_initial(f, Df, ef, x, c, max_solve_iterations)
    
    # Initialize trace
    trace = FiberTrace(c)
    trace.x = x
    trace.points.append(x)
    trace.residuals.append(initial_residuals[-1])
    
    # Traverse
    for step in it.count(0):

        # Update DF
        DF = np.concatenate((Df(x[:N])[0], -c), axis=1)
        
        # Update tangent
        z = compute_tangent(DF, z)
        
        # Update trace
        trace.DF = DF
        trace.z = z
        if step == 0: trace.z_initial = z

        # Get step size
        step_amount, step_data = compute_step_amount(trace)
        if max_step_size is not None:
            step_amount = np.sign(step_amount)*min(np.fabs(step_amount), max_step_size)
               
        # Update x
        x, step_residuals = take_step(f, Df, ef, c, z, x,
            step_amount, max_solve_iterations)

        # Log and store progress
        trace.x = x
        trace.points.append(x)
        trace.tangents.append(z)
        trace.residuals.append(step_residuals[-1])
        trace.step_amounts.append(step_amount)
        trace.step_data.append(step_data)
        if logger is not None and step % 10 == 0:
            logger.log("step %d: residual %.3f, theta %.3f, step data %s...\n"%(
                step, step_residuals[-1], step_amount, step_data))

        # Check for early termination criteria
        if max_traverse_steps is not None and step + 1 >= max_traverse_steps:
            trace.status = "Max steps"
            break
        if stop_time is not None and time.clock() >= stop_time:
            trace.status = "Timed out"
            break
        # Check custom termination criteria
        if terminate is not None and terminate(trace):
            trace.status = "Terminated"
            break
        # Check for closed loop
        if len(trace.points) > 2 and np.fabs(trace.points[-1]-trace.points[0]).max() < np.fabs(trace.points[2]-trace.points[0]).max():
            trace.status = "Closed loop"
            break
        
    # final output
    if logger is not None: logger.log("Status: %s\n"%trace.status)
    return trace

def plot_fiber(X, Y, V, f, ax=None, scale_XY=1, scale_V=1, fiber_color='k'):
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
    ax.plot(V[0,:],V[1,:],color=fiber_color, linestyle='-', linewidth=1)
    ax.quiver(V[0,:],V[1,:],C_V[0,:],C_V[1,:],color=0.0*np.ones((1,3)),
        scale=scale_V,units='xy',angles='xy')

    # Set limits based on grid and show
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    ax.set_xlim((X.min(),X.max()))
    ax.set_ylim((Y.min(),Y.max()))
