"""
Basic recurrent neural network model with activation rule:
    v[t+1] = np.tanh(W.dot(v[t]))
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import dfibers.numerical_utilities as nu
import dfibers.logging_utilities as lu
import dfibers.fixed_points as fx
import dfibers.traversal as tv
import dfibers.solvers as sv

def f_factory(W):
    """
    For a given weight matrix W, returns the function f,
    where f(V)[:,p] is the change in network state V[:,p] after one update
    """
    return lambda V: np.tanh(W.dot(V)) - V
    
def Df_factory(W):
    """
    For a given weight matrix W, returns the function Df,
    where Df(V)[p,:,:] is the Jacobian of f at V[:,[p]]
    """
    I = np.eye(W.shape[0])
    def Df(V):
        D = 1-np.tanh(W.dot(V))**2
        return D.T[:,:,np.newaxis]*W[np.newaxis,:,:] - I[np.newaxis,:,:]
    return Df

def ef_factory(W):
    """
    For a given weight matrix W, returns the function ef,
    where ef(V) is an upper bound on the forward error of f(V).
    Specifically, this should be true in every matrix entry:
        |{f({V})} - f(V)| < {ef({V})},
    where V is within machine precision of machine approximation {V}
    and {f({V})}, {ef({V})} are the computed machine approximations of f({V}), ef({V}).
    """
    e_sigma = 5
    W_abs = np.fabs(W)
    def ef(V):
        """
        Estimates the numerical forward error in numpy.tanh(W.dot(V))-V.
        Returns the numpy.array margin, where
          margin[i,j] == the forward error bound on (numpy.tanh(W.dot(V))-V)[i,j].
        """
        N = V.shape[0]
        V_eps = nu.eps(V)
        tWV_eps = nu.eps(np.tanh(np.dot(W,V)))
        margin = np.dot(W_abs, V_eps)
        margin += N*nu.eps(np.dot(W_abs, np.fabs(V)))
        margin += e_sigma * tWV_eps
        margin += V_eps
        margin += np.maximum(tWV_eps, V_eps)
        return margin
    return ef

def sampler_factory(W):
    N = W.shape[0]
    return lambda : 2*np.random.rand(N,1) - 1

def qg_factory(W):
    """
    For a given weight matrix W, returns the function qg, where
    qg(v) returns the objective |f|^2 (and its gradient)
    See (Barak and Sussillo 2013)
    qg conforms to the scipy.optimize.minimize signature first parameter
    """
    f, Df = f_factory(W), Df_factory(W)
    def qg(v):
        """
        v is a flat numpy array
        returns
            q: the objective |f|^2 at v (a scalar)
            g: the gradient of q at v (a flat numpy array)
        """
        fv, Dfv = f(v[:,np.newaxis]), Df(v[:,np.newaxis])[0]
        q, g = (fv**2).sum(), Dfv.T.dot(fv).flatten()
        return q, g
    return qg

def H_factory(W):
    """
    For a given weight matrix W, returns the function H, where
    H(v) returns the approximate Hessian of the objective |f|^2
    See (Barak and Sussillo 2013)
    H conforms to the scipy.optimize.minimize signature "hess" parameter
    """
    Df = Df_factory(W)
    def H(v):
        """
        Calculates the Hessian of |f|^2 at v
        """
        Dfv = Df(v[:,np.newaxis])[0]
        return Dfv.T.dot(Dfv)
    return H

def compute_step_amount_factory(W):
    """
    For a given weight matrix W, returns the function compute_step_amount,
    which returns a certified step size at a particular fiber point.
    The function signature is compute_step_amount(trace),
    where trace includes fields DF, and z:
    DF is the derivative of F(x), and z is the fiber tangent.
    compute_step_amount's first return value is the step amount
    compute_step_amount's second return value is the minimum singular value of Dg at x
    """    
    mu = np.sqrt(16./27.) * np.linalg.norm(W) * min(np.linalg.norm(W), np.sqrt((W*W).sum(axis=1)).max())
    def compute_step_amount(trace):
        Dg = np.concatenate((trace.DF, trace.z.T), axis=0)
        sv_min, low_rank = nu.minimum_singular_value(Dg)
        step_amount = sv_min / (4. * mu)
        return step_amount, sv_min, low_rank
    return compute_step_amount

def compute_step_amount_factory3(W):
    """
    For a given weight matrix W, returns the function compute_step_amount,
    which returns a certified step size at a particular fiber point.
    The function signature is compute_step_amount(trace),
    where trace includes fields DF, and z:
    DF is the derivative of F(x), and z is the fiber tangent.
    compute_step_amount's first return value is the step amount
    compute_step_amount's second return value is the minimum singular value of Dg at x
    """

    # f2
    N = W.shape[0]
    def f2(x):
        Wv = W.dot(x[:N])
        sig2 = np.fabs(2*np.tanh(Wv)*(1 - np.tanh(Wv)**2))
        return (sig2*np.fabs(W**2).max(axis=1)[:,np.newaxis]).max()

    # f3
    sig3 = 2 # max |tanh'''|
    f3 = sig3 * np.fabs(W**3).max()

    # step size
    return tv.compute_step_amount_factory(f2, f3)

def terminate_factory(W, c):
    """
    For a given weight matrix W and direction vector c, returns the function terminate,
    where terminate(trace) returns true if termination is acceptable at current fiber point.
    Uses the termination criterion from (Katz and Reggia 2017)
    W is the weight matrix (N by N numpy.array)
    c is the direction vector (N by 1 numpy.array)
    returns term, the bound on alpha past which no more fixed points will be found
    """
    D_bound = min(1, 1/np.linalg.norm(W,ord=2))
    a_bound = ((np.arctanh(np.sqrt(1 - D_bound)) + np.fabs(W).sum(axis=1))/np.fabs(W.dot(c))).max()
    return lambda trace: np.fabs(trace.x[-1]) > a_bound

def duplicates_factory(W):
    """
    Simple duplicates check justified by (Katz and Reggia 2017)
    """
    return lambda V, v: (np.fabs(V - v) < 2**-21).all(axis=0)

def make_known_fixed_points(N):
    """
    Constructs an N-dimensional rnn with up to N known fixed points
    returns
        f, Df, ef: rnn functions
        W: rnn weight matrix
        V[:,[k]]: the k^{th} known fixed point
    """
    # Sample random points
    V = 2.*np.random.rand(N,N) - 1.
    # Construct weight matrix known to have them as fixed points
    W = nu.mrdivide(np.arctanh(V), V)
    # Refine points to counteract finite-precision round-off error
    f, ef, Df = f_factory(W), ef_factory(W), Df_factory(W)
    V, fixed = fx.refine_points(V, f, ef, Df)
    return f, Df, ef, W, V[:,fixed]

def run_fiber_solver(W, **kwargs):
    """
    rnn-specific convenience wrapper for generic fiber solver and post-processing
    W is the (N,N) weight matrix; kwargs are as in dfibers.solvers.fiber_solver
    """

    # Setup random c for termination criteria
    N = W.shape[0]
    if "c" in kwargs:
        c = kwargs["c"]
    else:
        c = np.random.randn(N,1)
        c = c/np.linalg.norm(c)
        kwargs["c"] = c
    
    # Setup other parameters if not provided
    kwargs["v"] = kwargs.get("v", np.zeros((N,1)))
    kwargs["max_solve_iterations"] = kwargs.get("max_solve_iterations", 2**5)
    if "compute_step_amount" not in kwargs:
        kwargs["compute_step_amount"] = compute_step_amount_factory(W)

    # Run solver from origin
    solution = sv.fiber_solver(
        f = f_factory(W),
        ef = ef_factory(W),
        Df = Df_factory(W),
        terminate = terminate_factory(W, c),
        **kwargs
    )

    # Post-process fixed points
    fxpts = solution["Fixed points"]
    fxpts = np.concatenate((-fxpts, np.zeros((N,1)), fxpts), axis=1)
    fxpts = fx.sanitize_points(
        fxpts,
        f = f_factory(W),
        ef = ef_factory(W),
        Df = Df_factory(W),
        duplicates = duplicates_factory(W),
    )
    
    # Return post-processed fixed points and full solver result
    return fxpts, solution

if __name__ == "__main__":

    # Set up 2D network
    N = 2
    W = 1.25*np.eye(N) + 0.1*np.random.randn(N,N)

    logger = lu.Logger(sys.stdout)
    fxpts, solution = run_fiber_solver(W,
        # max_history=100,
        compute_step_amount = compute_step_amount_factory3(W),
        logger=logger,
        abs_alpha_min = True,
        within_fiber = True)

    # Extract steps along fiber and corresponding f(v)'s
    trace = solution["Fiber trace"]
    X = np.concatenate(trace.points, axis=1)
    X = np.concatenate((-np.fliplr(X), X), axis=1)
    V = X[:-1,:]

    # Plot fiber and fixed points
    X_grid, Y_grid = np.mgrid[-1.15:1.15:20j,-1.15:1.15:20j]
    plt.figure(figsize=(5,4.5))
    tv.plot_fiber(X_grid, Y_grid, V, f_factory(W), scale_XY=1, scale_V=1)
    plt.scatter(*fxpts, color='k', marker='o')
    plt.xlabel("v1")
    plt.ylabel("v2",rotation=0)
    plt.tight_layout()
    plt.show()
    
