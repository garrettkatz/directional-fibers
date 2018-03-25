import numpy as np
import matplotlib.pyplot as pt
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(linewidth=200,precision=4)


def R(v):
    return (100*(v[:-1,:]**2 - v[1:,:])**2 + (v[:-1,:] - 1)**2).sum(axis=0)

def f(v):
    return np.concatenate((
        400*v[[0],:]*(v[[0],:]**2 - v[[1],:]) + 2*(v[[0],:] - 1),
        -200*(v[:-2,:] - v[1:-1,:]) + 400*v[1:-1,:]*(v[1:-1,:]**2- v[2:,:]) + 2*(v[1:-1,:] - 1),
        -200*(v[[-2],:]**2 - v[[-1],:]),
    ), axis=0)

def ef(v):
    return 0.0001*np.ones(v.shape)

def Df(v):
    Dfv = np.zeros((v.shape[1], v.shape[0], v.shape[0]))
    for i in range(v.shape[0]-1):
        Dfv[:,i,i] = 200*(6*v[i,:]**2 - 2*v[i+1,:] + 0.01)
        Dfv[:,i,i+1] = -400*v[i,:]
        Dfv[:,i+1,i] = -400*v[i,:]
    Dfv[:,-1,-1] = 200
    return Dfv

def compute_step_amount(trace):
    return (0.000001, 0)

def sanity2D():

    # Set up fiber arguments
    v = np.array([[2],[.5]])
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": compute_step_amount,
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

    # Compute rosenbrock
    R_surface = R(np.array([X_surface.flatten(), Y_surface.flatten()]))
    R_surface = R_surface.reshape(X_surface.shape)

    # Visualize fiber and surface
    ax_fiber = pt.subplot(2,1,2)
    tv.plot_fiber(X_fiber, Y_fiber, V, f, ax=ax_fiber, scale_XY=500, scale_V=25)
    ax_fiber.plot([1],[1],'ko') # global optimum
    ax_surface = pt.gcf().add_subplot(2,1,1,projection="3d")
    ax_surface.plot_surface(X_surface, Y_surface, R_surface, linewidth=0, antialiased=False, color='gray')
    ax_surface.view_init(azim=-98, elev=21)
    for ax in [ax_fiber, ax_surface]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if ax == ax_surface:
            ax.set_zlabel("R(x,y)", rotation=90)
            ax.set_zlim([0,10000])
            ax.view_init(elev=40,azim=-106)
        ax.set_xlim([-2.5,3.5])
    pt.show()

def sanity3D():

    # Set up fiber arguments
    # v = np.ones((3,1)) + np.random.randn(3,1)*0.1
    v = np.zeros((3,1))
    c = f(v)
    # v = np.ones((3,1))
    # c = np.random.randn(*v.shape)
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": compute_step_amount,
        "v": v,
        "c": c,
        "max_step_size": 1,
        "max_traverse_steps": 120,
        "max_solve_iterations": 2**5,
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    trace = solution["Fiber trace"]
    # V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,::10]
    V1 = np.concatenate(trace.points, axis=1)[:-1,::10]
    z = trace.z_initial
    
    # # Run in other direction (negate initial tangent)
    # fiber_kwargs["z"] = -z
    # solution = sv.fiber_solver(**fiber_kwargs)
    # V2 = np.concatenate(trace.points, axis=1)[:-1,::10]

    # # Join fiber segments
    # V = np.concatenate((np.fliplr(V1), V2), axis=1)
    # C = f(V)
    # print(V)

    V = V1
    C = f(V)*0.0001
    print(np.sqrt((C*C).sum(axis=0)))
    # print(V)
    # print(trace.residuals)
    # print(trace.step_amounts)

    ax = pt.gca(projection='3d')
    ax.plot(*V,marker='o',color='k')
    ax.quiver(*np.concatenate((V, C),axis=0),color='k',arrow_length_ratio=0.00001)
    pt.show()

if __name__ == "__main__":

    # sanity2D()
    sanity3D()
