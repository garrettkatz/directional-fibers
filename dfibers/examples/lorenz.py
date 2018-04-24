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
    Dfv = np.empty((v.shape[1],3,3))
    Dfv[:,0,0], Dfv[:,0,1], Dfv[:,0,2] = -s, s, 0
    Dfv[:,1,0], Dfv[:,1,1], Dfv[:,1,2] = r - v[2], -1, -v[0]
    Dfv[:,2,0], Dfv[:,2,1], Dfv[:,2,2] = v[1], v[0], -b
    return Dfv

if __name__ == "__main__":

    # Collect attractor points
    t = np.arange(0,40,0.01)
    v = np.ones((N,1))
    A = si.odeint(lambda v, t: f(v.reshape((N,1))).flatten(), v.flatten(), t).T

    # Set up fiber arguments
    v = np.zeros((N,1))
    # c = np.random.randn(N,1)
    c = np.array([[0.83736021, -1.87848114, 0.43935044]]).T
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.1, 0, False),
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:N,:]) > 50).any(),
        "max_step_size": 1,
        "max_traverse_steps": 2000,
        "max_solve_iterations": 2**5,
    }
    print("using c:")
    print(c.T)

    # Visualize strange attractor
    ax = pt.gca(projection="3d")
    ax.plot(*A, color='gray', linestyle='-', alpha=0.5)
    br1 = np.sqrt(b*(r-1))
    U = np.array([[0, 0, 0],[br1,br1,r-1],[-br1,-br1,r-1]]).T
    ax.scatter(*U, color='black')

    # Run and visualize fiber components, for each fxpt
    xlims, ylims, zlims = [-20,20], [-30,30], [-20,60]
    for fc in [0,2]:

        # start from current fxpt
        fiber_kwargs["v"] = U[:,[fc]]
        # ax.text(U[0,fc],U[1,fc],U[2,fc], str(fc))

        # Run in one direction
        solution = sv.fiber_solver(**fiber_kwargs)
        V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]
        z = solution["Fiber trace"].z_initial

        # Run in other direction (negate initial tangent)
        fiber_kwargs["z"] = -z
        solution = sv.fiber_solver(**fiber_kwargs)
        V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]
    
        # Join fiber segments, restrict to figure limits
        V = np.concatenate((np.fliplr(V1), V2), axis=1)
        V = V[:,::50]
        for i, (lo, hi) in enumerate([xlims, ylims, zlims]):
            V = V[:,(lo < V[i,:]) & (V[i,:] < hi)]
        C = f(V)

        # Visualize fiber
        ax.plot(*V, color='black', linestyle='-')
        ax.quiver(*np.concatenate((V,.1*C),axis=0),color='black')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=15,azim=145)
    pt.tight_layout()
    pt.show()
