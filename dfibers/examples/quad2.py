"""
Fiber-based fixed point location in the 2-variable quadratic system
    f([x,y].T) = M @ [1, x, y, xy, x**2, y**2].T
"""

import numpy as np
import matplotlib.pyplot as pt
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

def f_factory(M):
    """
    For a given 2x6 coefficient matrix M, returns the function f,
    where f(V)[:,p] = f(V[:,p])
    """
    def f(V):
        x, y = V
        Q = np.stack([np.ones(len(x)), x, y, x*y, x**2, y**2])
        return M @ Q
    return f
    
def Df_factory(M):
    """
    For a given 2x6 coefficient matrix M, returns the function Df,
    where Df(V)[p,:,:] is the Jacobian of f at V[:,[p]]
    """
    def Df(V):
        x, y = V
        dQ = np.zeros((len(x), 6, 2))
        dQ[:,1,0] = 1
        dQ[:,3,0] = y
        dQ[:,4,0] = 2*x
        dQ[:,2,1] = 1
        dQ[:,3,1] = x
        dQ[:,5,1] = 2*y
        return M @ dQ
    return Df

def ef(v):
    """
    Simple forward error bound (TODO)
    """
    return 0.001*np.ones(v.shape) # placeholder

if __name__ == "__main__":

    # random quadric
    M = np.random.randn(2,6)
    f = f_factory(M)
    Df = Df_factory(M)

    # Set up fiber arguments
    # v = np.array([[-.5],[-1.5]])
    v = np.ones((2,1))
    c = f(v)
    # v = None
    # c = np.array([[1],[.25]])
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.01, 0, False),
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:2,:]) > 10).any(),
        "max_step_size": 1,
        "max_traverse_steps": 500,
        "max_solve_iterations": 2**5,
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,:]
    z = solution["Fiber trace"].z_initial
    
    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,:]

    # Join fiber segments
    V = np.concatenate((np.fliplr(V1), V2), axis=1)
    V = V[:,np.isfinite(V).all(axis=0)]
    V = V[:,(np.fabs(V) < 3).all(axis=0)]

    # Grids for fiber and attractor
    X_fiber, Y_fiber = np.mgrid[-2:2:20j, -2:2:20j]
    X_a, Y_a = np.mgrid[-1:1:100j, -1:1:100j]  

    # Visualize fiber and attractor
    pt.figure(figsize=(3.5,3.5))
    ax_fiber = pt.gca()
    tv.plot_fiber(X_fiber, Y_fiber, V[:,::10], f, ax=ax_fiber, scale_XY=10, scale_V=10)
    ax_fiber.set_xlabel("x")
    ax_fiber.set_ylabel("y",rotation=0)
    pt.yticks(np.linspace(-2,2,5))
    pt.tight_layout()
    pt.show()

