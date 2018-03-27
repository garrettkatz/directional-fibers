import numpy as np
import matplotlib.pyplot as pt
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(linewidth=200,precision=4)

TPI = 2*np.pi
RC = 1

def rastrigin(v):
    return (v**2 - RC*np.cos(TPI*v)).sum(axis=0)

def f(v):
    return 2*v + RC*TPI*np.sin(TPI*v)

def ef(v):
    return 0.0001*np.ones(v.shape)

def Df(v):
    Dfv = np.zeros((v.shape[1], v.shape[0], v.shape[0]))
    for i in range(v.shape[0]):
        Dfv[:,i,i] = 2 + RC*TPI**2*np.cos(TPI*v[i,:])
    return Dfv

def compute_step_amount(trace):
    mu = RC*TPI**3
    Dg = np.concatenate((trace.DF, trace.z.T), axis=0)
    sv_min = nu.minimum_singular_value(Dg)
    step_amount = sv_min / (4. * mu)
    return step_amount, sv_min
    # return (0.000001, 0)

if __name__ == "__main__":

    N = 2

    # Set up fiber arguments
    # v = np.ones((N,1)) + np.random.randn(N,1)*0.1
    v = np.random.randn(N,1)*0.1
    # v = np.ones((N,1))
    # v = np.zeros((N,1))
    c = f(v)
    # v = np.ones((N,1))
    # c = np.random.randn(*v.shape)
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.01,None), #compute_step_amount,
        "v": v,
        "c": c,
        "max_step_size": 1,
        "max_traverse_steps": 20000,
        "max_solve_iterations": 2**5,
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    trace = solution["Fiber trace"]
    # V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,::10]
    V1 = np.concatenate(trace.points, axis=1)[:-1,::100]
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
    C = f(V)
    # print(np.sqrt((C*C).sum(axis=0)))
    # print(V)
    # print(trace.residuals)
    print(trace.step_amounts[:10])
    print(trace.step_data[:10])

    pt.plot(*V,marker='o',color='k')
    pt.show()

    # ax = pt.gca(projection='3d')
    # ax.plot(*V,marker='o',color='k')
    # # ax.quiver(*np.concatenate((V, C*0.01),axis=0),color='k')
    # pt.show()


