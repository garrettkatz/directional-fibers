"""
Fiber-based optimization of 2D Rosenbrock function:
    rosenbrock(v) = (a - v[0])**2 + b*(v[1] - v[0]**2)**2
Traverses directional fibers of the gradient:
    f(v)[0] = -2*(a - v[0]) + -4*b*(v[1] - v[0]**2)*v[0]
    f(v)[1] = 2*b*(v[1] - v[0]**2)
"""

import numpy as np
import matplotlib.pyplot as pt
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

a, b = 1, 10

def f(v):
    return np.array([
        -2*(a - v[0,:]) + -4*b*(v[1,:] - v[0,:]**2)*v[0,:],
        2*b*(v[1,:] - v[0,:]**2),
    ])

def ef(v):
    return 0.001*np.ones((2,1))

def Df(v):
    Dfv = np.empty((v.shape[1],2,2))
    Dfv[:,0,0], Dfv[:,0,1] = 2 - 4*b*(v[1]-3*v[0]**2), -4*b*v[0]
    Dfv[:,1,0], Dfv[:,1,1] = -4*b*v[0], 2*b
    return Dfv

if __name__ == "__main__":

    # Set up fiber arguments
    v = np.array([[2],[.5]])
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.01, 0, False),
        "v": v,
        "c": f(v),
        "max_step_size": 1,
        "max_traverse_steps": 2000,
        "max_solve_iterations": 2**5,
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,::10]
    z = solution["Fiber trace"].z_initial
    
    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,::10]

    # Join fiber segments
    V = np.concatenate((np.fliplr(V1), V2), axis=1)

    # Grids for fiber and surface
    X_fiber, Y_fiber = np.mgrid[-2.5:3.5:40j, -1:4:40j]
    X_surface, Y_surface = np.mgrid[-2.5:3.5:100j, -1:4:100j]

    # Low rank Df curve
    x_lork = np.linspace(-2.5,3.5,40)
    y_lork = (2 + (12-8*b)*x_lork**2)/(4*b)

    # Compute rosenbrock
    R = (a - X_surface)**2 + b*(Y_surface - X_surface**2)**2

    # Visualize fiber and surface
    ax_fiber = pt.subplot(2,1,2)
    tv.plot_fiber(X_fiber, Y_fiber, V, f, ax=ax_fiber, scale_XY=500, scale_V=25)
    ax_fiber.plot([1],[1],'ko') # global optimum
    # ax_fiber.plot(x_lork, y_lork, 'r-') # low-rank Df points
    ax_surface = pt.gcf().add_subplot(2,1,1,projection="3d")
    ax_surface.plot_surface(X_surface, Y_surface, R, linewidth=0, antialiased=False, color='gray')
    ax_surface.view_init(azim=-98, elev=21)
    for ax in [ax_fiber, ax_surface]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if ax == ax_surface:
            ax.set_zlabel("R(x,y)", rotation=90)
            ax.set_zlim([0,1000])
            ax.view_init(elev=40,azim=-106)
        ax.set_xlim([-2.5,3.5])
    pt.show()
