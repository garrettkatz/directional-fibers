"""
Hopfield's analog neural network model:
    dv/dt = (T.dot(g(v)) - v/R + I)/C
    g(v) = (2/np.pi)*np.arctan(np.pi*L*v/2)
"""
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as pt
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.traversal as tv
import dfibers.solvers as sv

def f_factory(T, L, R, I, C):
    """
    For given network parameters, returns the function f,
    where f(V)[:,p] is the instantaneous change in network state V[:,p]
    """
    return lambda V: (T.dot((2/np.pi)*np.arctan(np.pi*L*V/2)) - V/R + I)/C
    
def Df_factory(T, L, R, I, C):
    """
    For a given network parameters, returns the function Df,
    where Df(V) is the derivative of f(V)
    if V has more than one column, Df(V)[p,:,:] is the derivative at the p^th one
    """
    def Df(V):
        D = (2/np.pi)*np.pi*L/2/(1 + (np.pi*L*V/2)**2)
        return (
            np.diagflat(C**-1).dot((T.dot(np.diagflat(D)) - np.diagflat(R**-1)))
        )[np.newaxis,:,:] # todo stack of multiple Dfv
    return Df

def ef_factory(T, L, R, I, C):
    N = T.shape[0]
    return lambda V: 0.001*np.ones((N,1))

if __name__ == "__main__":

    # Set network parameters
    N = 2
    T = np.array([[0, 1],[1, 0]])
    L = 1.4
    R = 1.25*np.ones((N,1))
    I = 0.0*np.random.randn(N,1)
    C = np.ones((N,1))
    
    f = f_factory(T,L,R,I,C)
    Df = Df_factory(T,L,R,I,C)
    ef = ef_factory(T,L,R,I,C)

    # Set up fiber arguments
    v = np.zeros((N,1))
    # c = np.random.randn(N,1)
    c = np.array([[1],[-.5]])
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.01, 0, False),
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:N,:]) > 10).any(),
        "max_step_size": 1,
        "max_traverse_steps": 2000,
        "max_solve_iterations": 2**5,
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]
    z = solution["Fiber trace"].z_initial
    
    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]

    # Join fiber segments
    V = np.concatenate((np.fliplr(V1), V2), axis=1)
    V = V[:,::20]

    # Visualize fiber and energy contours
    X_e, Y_e = np.mgrid[-2:2:40j,-2:2:40j]
    V_e = np.array([X_e.flatten(), Y_e.flatten()])
    G_e = (2/np.pi)*np.arctan(np.pi*L*V_e/2)
    E = np.diag(-.5*G_e.T.dot(T.dot(G_e)))
    E = E + (2/np.pi)**2/L * ((-np.log(1/np.sqrt(1 + (np.pi*L*V_e/2)**2)))/R).sum(axis=0)
    E = E + (I*G_e).sum(axis=0)
    E = E.reshape(X_e.shape)
    pt.contour(X_e, Y_e, E, [-.11, -.08, .1, .5, 1], colors='gray', linestyles='solid')
    X_f, Y_f = np.mgrid[-2:2:15j,-2:2:15j]
    tv.plot_fiber(X_f, Y_f, V, f, scale_XY=8, scale_V=3)
    
    pt.xlabel("x")
    pt.ylabel("y")
    pt.show()
