"""
Fiber-based nanophotonic structure design (toy example)
    a: curl operator
    x0: target field
    xi: resonance constant
Lagrangian of equality-constrained objective:
    L(v) = L(x,y,m) = (x - x0)**2 + m*(a**2 * y - xi*x)
    x is field, y is dielectric structure permittivity, m is multiplier
Vector field is gradient of L:
    f(v)[0] = 2*(v[0] - x0) + m*(a**2 * y - xi)
    f(v)[1] = m * a**2 * x
    f(v)[2] = a**2 * x * y - xi * x

Reference:
https://opg.optica.org/oe/fulltext.cfm?uri=oe-18-4-3793&id=195542
"""

import numpy as np
import matplotlib.pyplot as pt
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

rcParams["font.size"] = 14
rcParams["font.family"] = "serif"
rcParams['text.usetex'] = True

N = 3

# a = 1 + np.random.rand()
# x0 = 1 + np.random.rand()
# xi = .1*np.random.rand()
a= 1.0358980395014457
x0= 1.5675946100046785
xi=0.05310085202487887

print(f"A={a}, x0={x0}, xi={xi}")

def f(v):
    return np.stack([
        2*(v[0] - x0) + v[2]*(a**2 * v[1] - xi),
        v[2] * a**2 * v[0],
        a**2 * v[0] * v[1] - xi * v[0],
    ])

def ef(v):
    return 0.001*np.ones((N,1))

def Df(v):
    Dfv = np.empty((v.shape[1],3,3))
    Dfv[:,0,0], Dfv[:,0,1], Dfv[:,0,2] = 2,               v[2] * a**2, a**2 * v[1] - xi
    Dfv[:,1,0], Dfv[:,1,1], Dfv[:,1,2] = v[2] * a**2,     0,           a**2 * v[0]
    Dfv[:,2,0], Dfv[:,2,1], Dfv[:,2,2] = a**2 * v[1] - xi, a**2 * v[0], 0
    return Dfv

if __name__ == "__main__":

    # Set up fiber arguments
    # v = np.random.randn(N,1)
    # v = np.array([[ 1.57974222, -1.3565096,  -1.27208006]]).T
    v = np.array([[ 1.5, -1.3,  -1.2]]).T
    c = f(v)
    print("v,c:")
    print(v.T)
    print(c.T)

    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.01, 0, False),
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:N,:]) > 100).any(),
        "max_step_size": 1,
        "max_traverse_steps": 2000,
        "max_solve_iterations": 2**5,
    }

    fig, axes = pt.subplot_mosaic("AB",
         per_subplot_kw={('A',): {'projection': '3d'}},
         gridspec_kw={'width_ratios': [1.25, 1]},
         figsize=(5.5, 3))
         # figsize=(7, 3))

    # fig = pt.figure(figsize=(5.75,3))

    # plot fiber
    ax = axes["A"]
    ax.view_init(elev=20, azim=-105)

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]
    R1 = solution["Fixed points"]
    z = solution["Fiber trace"].z_initial

    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]
    R2 = solution["Fixed points"]

    # Join fiber segments, restrict to figure limits
    V = np.concatenate((np.fliplr(V1), V2), axis=1)
    V = V[:,::50]
    R = np.concatenate((R1, R2), axis=1)

    # # Truncate fiber
    # xlims, ylims, zlims = [-20,20], [-30,30], [-20,60]
    # for i, (lo, hi) in enumerate([xlims, ylims, zlims]):
    #     V = V[:,(lo < V[i,:]) & (V[i,:] < hi)]

    # Get constant vectors along fiber
    C = f(V)

    # Post-process fixed points
    duplicates = lambda U, v: (np.fabs(U - v) < 10e-4).all(axis=0)
    roots = fx.sanitize_points(R, f, ef, Df, duplicates)

    # check points
    print('roots, f(roots)')
    print(roots)
    print(f(roots))

    # Visualize fiber
    ax.plot(*V, color='black', linestyle='-')
    ax.quiver(*np.concatenate((V,.1*C),axis=0),color='black')
    ax.plot(*roots, marker='o', color='k', linestyle='none')

    # ax.set_xticks([],[])
    # ax.set_yticks([],[])
    # ax.set_zticks([],[])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("$\\lambda$", rotation=0)
    # ax.view_init(elev=15,azim=145)


    # visualize optimization problem
    # pt.subplot(1,2,2)

    ax = axes["B"]

    y0 = xi / a**2
    xlims = [-12, 5]
    ylims = [-3, 6]
    lim = 1.1*np.fabs(V[:2]).max()

    xs = np.linspace(*xlims, 50)
    ys = np.linspace(*ylims, 50)
    X, Y = np.meshgrid(xs, ys)
    Z = X * (a**2 * Y - xi)
    CS = pt.contour(X, Y, Z, levels=10)
    ax.clabel(CS, inline=True, fontsize=10)

    ax.plot([x0, x0], ylims, 'k--')
    # pt.plot(xlims, [y0, y0], 'k-.')

    ax.plot(*V[:2], color='black', linestyle='-')
    ax.plot(*roots[:2], marker='o', color='k', linestyle='none')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)

    pos = ax.get_position()
    pos = [pos.x0, pos.y0 + pos.height*.1, pos.width, pos.height*.9]
    ax.set_position(pos)

    # fig.suptitle("SA3")
    # pt.tight_layout()
    pt.show()
