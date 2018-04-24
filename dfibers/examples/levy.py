"""
Fiber-based optimization of Levy function:
    levy(v0,v1) = sin^2(pi w0) + (w0-1)^2 * (1 + 10 sin^2(pi*w0+1)) + (w1-1)^2(1 + sin^2(2pi*w1))
    wi = 1 + (vi-1)/4
"""

import sys
import numpy as np
import matplotlib.pyplot as pt
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.logging_utilities as lu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

def levy(v):
    w = 1 + (v-1)/4
    return np.sin(np.pi*w[0])**2 + (w[0]-1)**2 * (1 + 10*np.sin(np.pi*w[0]+1)**2) + \
        (w[1]-1)**2 *(1 + np.sin(2*np.pi*w[1])**2)

def f(v):
    w = 1 + (v-1)/4
    return np.array([
        np.sin(2*np.pi*w[0])*np.pi*.25 + \
        2*(w[0]-1)*.25*(1 + 10*np.sin(np.pi*w[0]+1)**2) + \
        (w[0]-1)**2 * 10*np.sin(2*(np.pi*w[0]+1))*np.pi*.25
        ,
        2*(w[1]-1)*.25*(1 + np.sin(2*np.pi*w[1])**2) + \
        (w[1]-1)**2* np.sin(2*2*np.pi*w[1])*2*np.pi*.25
        ])

def ef(v):
    return (10**-6)*np.ones(v.shape)

def Df(v):
    w = 1 + (v-1)/4
    Dfv = np.zeros((v.shape[1],v.shape[0],v.shape[0]))
    Dfv[:,0,0] = 2*np.pi*np.cos(2*np.pi*w[0])*.25*np.pi*.25 + \
        2*.25*.25*(1 + 10*np.sin(np.pi*w[0]+1)**2) + \
        2*(w[0]-1)*.25 *10*np.sin(2*(np.pi*w[0]+1))*np.pi*.25 + \
        2*(w[0]-1)*.25 *10*np.sin(2*(np.pi*w[0]+1))*np.pi*.25 + \
        (w[0]-1)**2 * 10*np.cos(2*(np.pi*w[0]+1))*2*np.pi*.25*np.pi*.25
    Dfv[:,1,1] = 2*.25*.25*(1 + np.sin(2*np.pi*w[1])**2) + \
        2*(w[1]-1)*.25*np.sin(2*2*np.pi*w[1])*2*np.pi*.25 + \
        2*(w[1]-1)*.25*np.sin(2*2*np.pi*w[1])*2*np.pi*.25 + \
        (w[1]-1)**2* np.cos(2*2*np.pi*w[1])*2*2*np.pi*.25*2*np.pi*.25
    return Dfv


if __name__ == "__main__":
    
    X_surface, Y_surface = np.mgrid[-10:10:100j,-10:10:100j]
    
    L = levy(np.array([X_surface.flatten(), Y_surface.flatten()])).reshape(X_surface.shape)
    
    ax_surface = pt.gcf().add_subplot(2,1,1,projection="3d")
    ax_surface.plot_surface(X_surface, Y_surface, L, linewidth=0, antialiased=False, color='gray')

    # Set up fiber arguments
    # v = 0.*np.ones((2,1)) + 1*np.random.randn(2,1)
    v = 20*np.random.rand(2,1) - 10
    c = f(v)
    c = c + 0.1*np.random.randn(2,1)
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.001, 0, False),
        "v": v,
        "c": c,
        "logger": lu.Logger(sys.stdout),
        "max_step_size": 1,
        "terminate": lambda trace: (np.fabs(trace.x[:-1]) > 100).any(),
        "max_traverse_steps": 20000,
        "max_solve_iterations": 2**5,
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,::10]
    FX1 = solution["Fixed points"]
    z = solution["Fiber trace"].z_initial
    print("Status: %s\n"%solution["Fiber trace"].status)    
    
    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,::10]
    FX2 = solution["Fixed points"]
    print("Status: %s\n"%solution["Fiber trace"].status)    

    # Join fiber segments
    V = np.concatenate((np.fliplr(V1), V2), axis=1)
    FX = np.concatenate((FX1, FX2), axis=1)

    X_grid, Y_grid = np.mgrid[-10:10:60j,-10:10:60j]
    ax = pt.gcf().add_subplot(2,1,2)
    tv.plot_fiber(X_grid, Y_grid, V[:,::20], f, ax=ax, scale_XY=10, scale_V=20)
    pt.plot(FX[0],FX[1], 'ko')

    pt.show()
    
    # v = np.zeros((2,1))
    # print(levy(v))
    # print(f(v).T)
    # v = np.ones((2,1))
    # print(levy(v))
    # print(f(v).T)
