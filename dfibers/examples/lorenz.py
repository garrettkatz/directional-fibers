"""
Fiber-based fixed point location in the Lorenz system
    f(v)[0] = s*(v[1]-v[0])
    f(v)[1] = r*v[0] - v[1] - v[0]*v[2]
    f(v)[2] = v[0]*v[1] - b*v[2]

Reference:
http://www.emba.uvm.edu/~jxyang/teaching/Math266notes13.pdf
https://en.wikipedia.org/wiki/Lorenz_system
"""

import numpy as np
import matplotlib.pyplot as pt
import scipy.integrate as si
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

N = 3
s, b, r = 10, 8./3., 28

def f(v):
    return np.array([
        s*(v[1,:]-v[0,:]),
        r*v[0,:] - v[1,:] - v[0,:]*v[2,:],
        v[0,:]*v[1,:] - b*v[2,:],
    ])

def ef(v):
    return 0.001*np.ones((N,1))

def Df(v):
    return np.array([
        [-s, s, 0],
        [r-v[2,0], -1, -v[0,0]],
        [v[1,0], v[0,0], -b],
    ])

if __name__ == "__main__":

    # Collect attractor points
    attractor = []
    t = np.arange(0,40,0.01)
    for s in range(5):
        v = 2*np.random.rand(N,1) - 1
        V = si.odeint(lambda v, t: f(v.reshape((N,1))).flatten(), v.flatten(), t)
        attractor.append(V.T)
    
    # Set up fiber arguments
    v = np.zeros((N,1))
    c = np.random.randn(N,1)
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.1, 0),
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:N,:]) > 50).any(),
        "max_step_size": 1,
        "max_traverse_steps": 2000,
        "max_solve_iterations": 2**5,
        "solve_tolerance": 10**-10,
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = solution["Fiber"]["X"][:N,:]
    z = solution["Fiber"]["z"]
    
    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = solution["Fiber"]["X"][:N,:]

    # Join fiber segments
    V = np.concatenate((np.fliplr(V1), V2), axis=1)
    V = V[:,::50]
    C = f(V)

    # Visualize fiber and strange attractor
    ax = pt.gca(projection="3d")
    ax.plot(*V, color='black', linestyle='-')
    ax.quiver(*np.concatenate((V,.1*C),axis=0),color='black')
    for a in attractor:
        ax.plot(*a, color='gray', linestyle='-', alpha=0.5)
    br1 = np.sqrt(b*(r-1))
    U = np.array([[0, 0, 0],[br1,br1,r-1],[-br1,-br1,r-1]]).T
    ax.scatter(*U, color='black')
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    pt.show()
