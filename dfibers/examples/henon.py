"""
Fiber-based fixed point location in the Henon map
    f(v)[0] = -v[0] + (1 - a*v[0]**2 + v[1])
    f(v)[1] = -v[1] + (b*v[0])
"""

import numpy as np
import matplotlib.pyplot as pt
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

a, b = 1.4, 0.3

def f(v):
    return np.array([
        1 - a*v[0,:]**2 - v[1,:] - v[0,:],
        b*v[0,:] - v[1,:],
    ])

def ef(v):
    return 0.001*np.ones((2,1))

def Df(v):
    Dfv = np.empty((v.shape[1],2,2))
    Dfv[:,0,0], Dfv[:,0,1] = -2*a*v[0] - 1, -1
    Dfv[:,1,0], Dfv[:,1,1] = b, -1
    return Dfv

if __name__ == "__main__":

    # # Collect attractor points
    # attractor = []
    # t = np.arange(0,40,0.01)
    # for s in range(5):
    #     v = 2*np.random.rand(N,1) - 1
    #     V = si.odeint(lambda v, t: f(v.reshape((N,1))).flatten(), v.flatten(), t)
    #     attractor.append(V.T)

    # Set up fiber arguments
    v = np.array([[-.5],[-1.5]])
    # v = np.ones((2,1))
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

    # Low rank Df curve
    x_lork = (b+1)/(-2*a)*np.ones(2)
    y_lork = np.array([-2, 2])
    
    # Fixed points
    x_fx = ((b+1) + np.array([-1, 1])*np.sqrt((b+1)**2 + 4*a))/(-2*a)
    y_fx = b*x_fx

    # Grids for fiber and attractor
    X_fiber, Y_fiber = np.mgrid[-2:2:20j, -2:2:20j]
    X_a, Y_a = np.mgrid[-1:1:100j, -1:1:100j]

    # Compute attractor points
    V_a = np.array([X_a.flatten(), Y_a.flatten()])
    for u in range(11):
        V_a = V_a + f(V_a)
    V_a = V_a[:,np.isfinite(V_a).all(axis=0)]
    V_a = V_a[:,(np.fabs(V_a) < 2).all(axis=0)]    

    # Visualize fiber and attractor
    pt.figure(figsize=(3.5,3.5))
    ax_fiber = pt.gca()
    ax_fiber.scatter(*V_a, marker='o', s=2, color=((0.3, 0.3, 0.3),)) # attractor
    tv.plot_fiber(X_fiber, Y_fiber, V[:,::10], f, ax=ax_fiber, scale_XY=10, scale_V=10)
    ax_fiber.plot(x_fx, y_fx, 'ko') # fixed points
    # ax_fiber.plot(x_lork, y_lork, 'r-') # low-rank Df points
    ax_fiber.set_xlabel("x")
    ax_fiber.set_ylabel("y",rotation=0)
    pt.yticks(np.linspace(-2,2,5))
    pt.tight_layout()
    pt.show()
