"""
Quadratic-term analog neural network model with fixed points at true corners
"""
import sys
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as pt
import dfibers.numerical_utilities as nu
import dfibers.logging_utilities as lu
import dfibers.fixed_points as fx
import dfibers.traversal as tv
import dfibers.solvers as sv

def f_factory(W):
    """
    For given network parameters, returns the function f,
    where f(V)[:,p] is the instantaneous change in network state V[:,p]
    """
    return lambda V: (W.dot(V))*(1 - V**2)
    
def Df_factory(W):
    """
    For a given weight matrix W, returns the function Df,
    where Df(V)[p,:,:] is the Jacobian of f at V[:,[p]]
    """
    def Df(V):
        DfV = (1-V**2).T[:,:,np.newaxis]*W[np.newaxis,:,:]
        WV2V = W.dot(V)*2*V
        for i in range(V.shape[0]):
            DfV[:,i,i] -= WV2V[i,:]
        return DfV
    return Df

def ef_factory(W):
    # return lambda V: 10**-6 * np.ones(V.shape)
    def ef(V):
        e1V2 = nu.eps(1-V**2) + \
            2*np.fabs(V)*nu.eps(V) + nu.eps(V**2) # error in 1-V**2
        eWV = V.shape[0]*nu.eps(np.fabs(W).dot(np.fabs(V))) + \
            np.fabs(W).dot(nu.eps(V)) # error in W.dot(V)        
        return nu.eps(W.dot(V)*(1-V**2)) + \
            np.fabs(W.dot(V))*e1V2 + \
            np.fabs(1-V**2)*eWV + \
            e1V2*eWV             
    return ef

def compute_step_amount_factory(W):

    # bound on |d**2f_i(v) / dv_j dv_k| for all i, j, k
    def f2(v):
        return 4*np.fabs(v).max()*np.fabs(W).max() + 2*np.fabs(W.dot(v)).max()

    # bound on |d**3f_i(v) / dv_j dv_k dv_l| for all v, i, j, k
    f3 = 6*np.fabs(W).max()

    # certified step size computation
    return tv.compute_step_amount_factory(f2, f3)

if __name__ == "__main__":

    # Set network parameters
    N = 4
    P = 2
    V = np.sign(np.random.randn(N,P))
    W = V.dot(V.T) - P*np.eye(N,N)
    
    f = f_factory(W)
    Df = Df_factory(W)
    ef = ef_factory(W)
    compute_step_amount = compute_step_amount_factory(W)

    # Set up fiber arguments
    v = np.zeros((N,1))
    # c = np.random.randn(N,1)
    c = V.dot(np.random.randn(P,1)) # span of training data
    logfile = sys.stdout
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": compute_step_amount,
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:N,:]) > 3).any(),
        "max_traverse_steps": 2000,
        "max_solve_iterations": 2**5,
        "max_history": 1000,
        "logger": lu.Logger(logfile)
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]
    V1_fx = solution["Fixed points"]
    z = solution["Fiber trace"].z_initial
    
    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]
    V2_fx = solution["Fixed points"]

    # Join fiber segments and fixed points
    V = np.concatenate((np.fliplr(V1), V2), axis=1)
    V_fx = np.concatenate((V1_fx, V2_fx), axis=1)
    
    fxpts = fx.sanitize_points(
        V_fx, f, ef, Df,
        duplicates = lambda V, v: (np.fabs(V - v) < 10**-6).all(axis=0),
    )
    
    print("Fxpts:")
    print(fxpts)
    print("Residuals:")
    print(ef(fxpts))
    
    # Visualize fiber if 2d
    if N == 2:
        X_f, Y_f = np.mgrid[-2:2:15j,-2:2:15j]
        V = V[:, \
            (V[0,:] >= X_f.min()) & \
            (V[0,:] <= X_f.max()) & \
            (V[1,:] >= Y_f.min()) & \
            (V[1,:] <= Y_f.max())]
        # V = V[:,::20]
        tv.plot_fiber(X_f, Y_f, V, f, scale_XY=8, scale_V=5)
    
        pt.plot(*fxpts, marker='o', color='k', linestyle='none')
        
        pt.xlabel("x")
        pt.ylabel("y")
        pt.show()
