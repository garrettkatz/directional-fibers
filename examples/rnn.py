"""
Basic recurrent neural network model with activation rule:
    v[t+1] = np.tanh(W.dot(v[t]))
"""
import numpy as np
import numerical_utilities as nu

def f_factory(W):
    """
    For a given weight matrix W, returns the function f,
    where f(v) is the change in network state v after one update
    """
    return lambda v: np.tanh(W.dot(v)) - v
    
def Df_factory(W):
    """
    For a given weight matrix W, returns the function Df,
    where Df(v) is the derivative of f(v)
    """
    I = np.eye(W.shape[0])
    return lambda v: (1-np.tanh(W.dot(v))**2)*W - I

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

