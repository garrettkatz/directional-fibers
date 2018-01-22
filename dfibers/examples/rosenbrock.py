"""
Fiber-based optimization of 2D Rosenbrock function:
    rosenbrock(v) = (a - v[0])**2 + b*(v[1] - v[0]**2)**2
Traverses directional fibers of the gradient:
    f(v)[0] = -2*(a - v[0]) + -4*b*(v[1] - v[0]**2)*v[0]
    f(v)[1] = 2*b*(v[1] - v[0]**2)
"""

import numpy as np
import matplotlib.pyplot as plt
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

a, b = 1, 100

def f(v):
    return np.array([
        [-2*(a - v[0]) + -4*b*(v[1] - v[0]**2)*v[0]],
        [2*b*(v[1] - v[0]**2)],
    ])

def ef(v):
    return 0.001*np.ones((2,1))

def Df(v):
    return np.array([
        [2 - 4*b*(v[1]-3*v[0]**2), -4*b*v[0]],
        [-4*b*v[0], 2*b],
    ])

if __name__ == "__main__":

    # Run solver from origin
    solution = sv.fiber_solver(
        f = f,
        ef = ef,
        Df = Df,
        compute_step_amount = lambda x, DF, z: tuple(0.0001, 0),
        N = 2,
        max_step_size = 1,
        max_traverse_steps = 50,
        max_solve_iterations = 2**5,
        solve_tolerance = 10**-18,
    )
    V = solution["Fiber"]["X"][:-1,:]
    C = f(V)

    a, b = 1, 100
    xy = np.mgrid[-2:2:30j, -1:3:30j]
    x, y = xy[0], xy[1]
    z = (a - x)**2 + b*(y - x**2)**2

    plt.subplot(1,2,1)
    plt.contour(x, y, z, 20)
    plt.plot(V[0,:],V[1,:],'b-')
    plt.gca().quiver(V[0,lm_idx],V[1,lm_idx],C[0,lm_idx],C[1,lm_idx],scale=.05,units='dots',width=2,headwidth=5)
    ax = plt.gcf().add_subplot(1,2,2,projection="3d")
    ax.plot_surface(x,y,z)
    plt.show()
