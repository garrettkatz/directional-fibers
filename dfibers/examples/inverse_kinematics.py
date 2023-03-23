import numpy as np
import matplotlib as mp
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import Axes3D

import dfibers.solvers as sv
import dfibers.traversal as tv
import dfibers.fixed_points as fx
from dfibers.logging_utilities import Logger

d = np.array([1, 1.25, 0])
target = np.ones((3,1))

def fk(v):
    c, s = np.cos(v), np.sin(v)
    elbow = np.array([
        c[1]*d[1],
        c[2]*s[1]*d[1],
        s[2]*s[1]*d[1],
    ])
    effector = np.array([
        c[1]*(c[0]*d[0] + d[1]) - s[1]*s[0]*d[0],
        c[2]*(s[1]*(c[0]*d[0] + d[1]) + c[1]*s[0]*d[0]),
        s[2]*(s[1]*(c[0]*d[0] + d[1]) + c[1]*s[0]*d[0]),
    ])
    return elbow, effector

def f(v):
    _, pos = fk(v)
    return (pos - target)

def Df(v):
    c, s = np.cos(v), np.sin(v)
    Dfv = np.zeros((v.shape[1], 3, 3))
    Dfv[:,0,0] = -c[1]*s[0]*d[0] - s[1]*c[0]*d[0]
    Dfv[:,0,1] = -s[1]*(c[0]*d[0]+d[1]) - c[1]*s[0]*d[0]
    Dfv[:,1,0] = c[2]*(-s[1]*s[0]*d[0] + c[1]*c[0]*d[0])
    Dfv[:,2,0] = s[2]*(-s[1]*s[0]*d[0] + c[1]*c[0]*d[0])
    Dfv[:,1,1] = c[2]*(c[1]*(c[0]*d[0]+d[1]) - s[1]*s[0]*d[0])
    Dfv[:,2,1] = s[2]*(c[1]*(c[0]*d[0]+d[1]) - s[1]*s[0]*d[0])
    Dfv[:,1,2] = -s[2]*(s[1]*(c[0]*d[0]+d[1]) + c[1]*s[0]*d[0])
    Dfv[:,2,2] = c[2]*(s[1]*(c[0]*d[0]+d[1]) + c[1]*s[0]*d[0])
    return Dfv

def ef(v): return 0.001

if __name__ == "__main__":

    mp.rcParams['font.family'] = 'serif'
    mp.rcParams['text.usetex'] = True

    # get initial point and fiber
    v = 0.05*np.ones((3,1))
    c = f(v)
    
    # Set up fiber arguments
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.001, 0, False),
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:2,:]) > np.pi).any(),
        "max_step_size": 1,
        "max_traverse_steps": 10000,
        "max_solve_iterations": 2**5,
        # "logger": logger,
    }
    
    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    X1 = np.concatenate(solution["Fiber trace"].points, axis=1)
    V1 = X1[:-1,:]
    A1 = X1[-1,:]
    R1 = solution["Fixed points"]
    z = solution["Fiber trace"].z_initial
    
    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    X2 = np.concatenate(solution["Fiber trace"].points, axis=1)
    V2 = X2[:-1,:]
    A2 = X2[-1,:]
    R2 = solution["Fixed points"]
    
    # Join fiber segments and roots
    V = np.concatenate((np.fliplr(V1), V2), axis=1)
    A = np.concatenate((A1[::-1], A2), axis=0)
    R = np.concatenate((R1, R2), axis=1)    
    C = f(V)

    duplicates = lambda U, v: (np.fabs(U - v) < 0.1).all(axis=0)
    R = fx.get_unique_points(R, duplicates)

    # remove spurious points
    elbow, effector = fk(R)
    R = R[:, np.fabs(effector - target).max(axis=0) < 0.01]
    print(f"{R.shape[1]} roots")

    pt.figure(figsize=(6.5, 3))

    ax = pt.subplot(1,2,1,projection="3d")
    pt.plot(*V, linestyle='-', color='k')
    pt.quiver(*np.concatenate((V[:,::100],.1*C[:,::100]),axis=0),color='black')
    pt.plot(*R, linestyle='none', marker='o', color='k')
    ax.set_title("Directional Fiber")
    ax.set_xlabel("$\\theta_0$", rotation=0)
    ax.set_ylabel("$\\theta_1$", rotation=0)
    ax.set_zlabel("$\\theta_2$", rotation=0)

    ax = pt.subplot(1,2,2,projection="3d")
    for j in range(0, V.shape[1], 500):
        elbow, effector = fk(V[:,j:j+1])
        pt.plot(*np.concatenate((np.zeros((3,1)), elbow, effector), axis=1), linestyle='-', color=(.75,)*3)
    for r in range(R.shape[1]):
        elbow, effector = fk(R[:,r:r+1])
        pt.plot(*np.concatenate((np.zeros((3,1)), elbow, effector), axis=1), linestyle='-', color='k')
    pt.plot(*target, marker='o', color='k')
    ax.set_title("Forward Kinematics")
    ax.set_xlabel("$x$", rotation=0)
    ax.set_ylabel("$y$", rotation=0)
    ax.set_zlabel("$z$", rotation=0)

    pt.tight_layout()
    pt.savefig("ik.eps")
    pt.show()


