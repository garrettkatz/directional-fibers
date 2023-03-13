"""
Fiber-based roots of a small power flow study
"""

import sys
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as pt
import dfibers.solvers as sv
import dfibers.traversal as tv
import dfibers.fixed_points as fx
from dfibers.logging_utilities import Logger

# http://www.ee.unlv.edu/~eebag/740%20Power%20Flow%20Analysis.pdf, pp 44-
def f(v):
    ang, mag = v
    return np.array([
        mag * 10 * np.sin(ang) + 2.0,
        mag * -10 * np.cos(ang) + mag**2 * 10 + 1.0,
    ])

def Df(v):
    ang, mag = v
    Dfv = np.empty((v.shape[1],2,2))
    Dfv[:,0,0] = mag * 10 * np.cos(ang)
    Dfv[:,0,1] = 10 * np.sin(ang)
    Dfv[:,1,0] = mag * 10 * np.sin(ang)
    Dfv[:,1,1] = -10 * np.cos(ang) + 2*mag * 10
    return Dfv

def ef(v):
    return 0.001*np.ones((2,1))

if __name__ == "__main__":

    mp.rcParams['font.family'] = 'serif'
    mp.rcParams['text.usetex'] = True

    logger = Logger(sys.stdout)

    # get initial point and fiber
    v = np.array([[0., 1.]]).T # "flat start"
    # v += np.random.randn(2,1) * 0.01
    c = f(v)

    # Set up fiber arguments
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.01, 0, False),
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:2,:]) > 10).any(),
        "max_step_size": 1,
        "max_traverse_steps": 1500,
        "max_solve_iterations": 2**5,
        # "logger": logger,
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,:]
    R1 = solution["Fixed points"]
    z = solution["Fiber trace"].z_initial

    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,:]
    R2 = solution["Fixed points"]

    # Join fiber segments and roots
    V = np.concatenate((np.fliplr(V1), V2), axis=1)
    R = np.concatenate((R1, R2), axis=1)

    xlo, xhi = V[0].min() - .1, V[0].max() + .1
    # xlo, xhi = -np.pi, np.pi
    ylo, yhi = V[1].min() - .1, V[1].max() + .1

    # Post-process fixed points
    duplicates = lambda U, v: (np.fabs(U - v) < 10e-4).all(axis=0)
    roots = fx.sanitize_points(R, f, ef, Df, duplicates)

    # Visualize fiber and roots
    X_fiber, Y_fiber = np.mgrid[xlo:xhi:20j, ylo:yhi:20j]
    pt.figure(figsize=(3.5,3.5))
    ax_fiber = pt.gca()
    tv.plot_fiber(X_fiber, Y_fiber, V[:,::10], f, ax=ax_fiber, scale_XY=None, scale_V=None)
    # ax_fiber.plot(x_fx, y_fx, 'ko') # fixed points
    # ax_fiber.plot(x_lork, y_lork, 'r-') # low-rank Df points
    ax_fiber.plot(*roots, marker='o', color='k', linestyle='none')
    ax_fiber.set_xlabel("$\\theta$ (rad)", fontsize=16)
    ax_fiber.set_ylabel("$R$", rotation=0, fontsize=16)
    # pt.yticks(np.linspace(-2,2,5))
    pt.tight_layout()
    pt.show()



