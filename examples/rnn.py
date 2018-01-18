"""
Basic recurrent neural network model with activation rule:
    v[t+1] = np.tanh(W.dot(v[t]))
"""
import numpy as np
import numerical_utilities as nu
import fixed_points as fx

def f_factory(W):
    """
    For a given weight matrix W, returns the function f,
    where f(V)[:,p] is the change in network state V[:,p] after one update
    """
    return lambda V: np.tanh(W.dot(V)) - V
    
def Df_factory(W):
    """
    For a given weight matrix W, returns the function Df,
    where Df(V) is the derivative of f(V)
    if V has more than one column, Df(V)[p,:,:] is the derivative at the p^th one
    """
    I = np.eye(W.shape[0])
    def Df(V):
        D = 1-np.tanh(W.dot(V))**2
        if V.shape[1] == 1: return D*W - I
        else: return D.T[:,:,np.newaxis]*W[np.newaxis,:,:] - I[np.newaxis,:,:]
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
        fv, Dfv = f(v[:,np.newaxis]), Df(v[:,np.newaxis])
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
        Dfv = Df(v[:,np.newaxis])
        return Dfv.T.dot(Dfv)
    return H

def compute_step_amount_factory(W):
    """
    For a given weight matrix W, returns the function compute_step_amount,
    which returns a certified step size at a particular fiber point.
    The function signature is compute_step_amount(x, DF, z),
    where x is the fiber point, the DF is the derivative of F(x), and z is the fiber tangent.
    compute_step_amount's second return value is the minimum singular value of Dg at x
    """    
    mu = np.sqrt(16./27.) * np.linalg.norm(W) * min(np.linalg.norm(W), np.sqrt((W*W).sum(axis=1)).max())
    def compute_step_amount(x, DF, z):
        Dg = np.concatenate((DF, z.T), axis=0)
        sv_min = nu.minimum_singular_value(Dg)
        step_amount = sv_min / (4. * mu)
        return step_amount, sv_min
    return compute_step_amount

def terminate_factory(W, c):
    """
    For a given weight matrix W and direction vector c, returns the function terminate,
    where terminate(x) returns true if termination is acceptable at point x.
    Uses the termination criterion from (Katz and Reggia 2017)
    W is the weight matrix (N by N numpy.array)
    c is the direction vector (N by 1 numpy.array)
    returns term, the bound on alpha past which no more fixed points will be found
    """
    D_bound = min(1, 1/np.linalg.norm(W,ord=2))
    a_bound = ((np.arctanh(np.sqrt(1 - D_bound)) + np.fabs(W).sum(axis=1))/np.fabs(W.dot(c))).max()
    return lambda x: np.fabs(x[-1]) > a_bound

def duplicates_factory(W):
    """
    Simple duplicates check justified by (Katz and Reggia 2017)
    """
    return lambda V, v: (np.fabs(V - v) < 2**-21).all(axis=0)

def make_known_fixed_points(N):

    # Sample random points
    V = 2.*np.random.rand(N,N) - 1.
    # Construct weight matrix known to have them as fixed points
    W = nu.mrdivide(np.arctanh(V), V)
    # Refine points to counteract finite-precision round-off error
    f, ef, Df = f_factory(W), ef_factory(W), Df_factory(W)
    V, fixed = fx.refine_points(V, f, ef, Df)
    return W, V[:,fixed]

