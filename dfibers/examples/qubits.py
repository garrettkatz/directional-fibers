import numpy as np
import torch as tr
import matplotlib as mp
import matplotlib.pyplot as pt

import dfibers.solvers as sv
import dfibers.traversal as tv
import dfibers.fixed_points as fx
from dfibers.logging_utilities import Logger

np.set_printoptions(linewidth=1000)

pauli = tr.tensor([
    [[1., 0.],
     [0., 1.]],

    [[0., 1.],
     [1., 0.]],

    [[0., -1j],
     [1j,  0.]],

    [[1., 0.],
     [0., -1.]],
])

pauli_prods = []
for a in range(4):
    for b in range(1,4):
        # if (a,b) in ((0,1), (0,2), (1,2)): continue
        if (a,b) in ((0,1), (0,2), (1,1)): continue
        print(a,b)
        pauli_prods.append(pauli[a] @ pauli[b])
pauli_prods = tr.stack(pauli_prods)

print(pauli_prods)

def get_spectrum(c):
    # c is (K, 9) tensor, K is batch size
    H = (pauli_prods[None,:,:,:] * c[:,:,None,None]).sum(dim=1) # (K, 2, 2)
    spectrum = tr.linalg.eigvalsh(H) # (K, 2)
    return spectrum

c0 = tr.randn(1,9)
target_spectrum = get_spectrum(c0)

def get_loss(c):
    # c is (K, 9) tensor, K is batch size
    spectrum = get_spectrum(c)
    return tr.sum((spectrum - target_spectrum)**2, dim=1) # (K,)

def f(v):
    # v is (9, K) numpy array, return (9, K) batch of gradients
    c = tr.tensor(v.T, requires_grad=True) # torch wants batch first
    loss = get_loss(c)
    loss.sum().backward()
    print(c.grad.numpy())
    return c.grad.numpy().T

def Df(v):
    # v is (9, K) numpy array, return (K, 9, 9) batch of hessians (jacobian of gradients)
    hess_fun = tr.func.hessian(get_loss)
    Dfv = []
    for k in range(v.shape[1]):
        c = tr.tensor(v[:,k:k+1].T)
        Dfv.append(hess_fun(c).squeeze()) # squeeze singleton batch dimensions
    Dfv = tr.stack(Dfv)
    print(tr.linalg.matrix_rank(Dfv))
    return Dfv.numpy()

def ef(v): return 0.001

if __name__ == "__main__":

    mp.rcParams['font.family'] = 'serif'
    mp.rcParams['text.usetex'] = True

    # get initial point and fiber
    v = c0.numpy().T
    
    # initial v is solution and f(v) = 0 there, so use default choice for constant direction vector (random)
    c = None # constant direction, not hamiltonian coefficients
    
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

    # pt.figure(figsize=(6.5, 3))

    # ax = pt.subplot(1,2,1,projection="3d")
    # pt.plot(*V, linestyle='-', color='k')
    # pt.quiver(*np.concatenate((V[:,::100],.1*C[:,::100]),axis=0),color='black')
    # pt.plot(*R, linestyle='none', marker='o', color='k')
    # ax.set_title("Directional Fiber")
    # ax.set_xlabel("$\\theta_0$", rotation=0)
    # ax.set_ylabel("$\\theta_1$", rotation=0)
    # ax.set_zlabel("$\\theta_2$", rotation=0)

    # ax = pt.subplot(1,2,2,projection="3d")
    # for j in range(0, V.shape[1], 500):
    #     elbow, effector = fk(V[:,j:j+1])
    #     pt.plot(*np.concatenate((np.zeros((3,1)), elbow, effector), axis=1), linestyle='-', color=(.75,)*3)
    # for r in range(R.shape[1]):
    #     elbow, effector = fk(R[:,r:r+1])
    #     pt.plot(*np.concatenate((np.zeros((3,1)), elbow, effector), axis=1), linestyle='-', color='k')
    # pt.plot(*target, marker='o', color='k')
    # ax.set_title("Forward Kinematics")
    # ax.set_xlabel("$x$", rotation=0)
    # ax.set_ylabel("$y$", rotation=0)
    # ax.set_zlabel("$z$", rotation=0)

    # pt.tight_layout()
    # pt.savefig("qubits.eps")
    # pt.show()




