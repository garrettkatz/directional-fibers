import sys
import itertools as it
import pickle as pk
import numpy as np
import torch as tr
import matplotlib as mp
import matplotlib.pyplot as pt

import dfibers.solvers as sv
import dfibers.traversal as tv
import dfibers.fixed_points as fx
from dfibers.logging_utilities import Logger

np.set_printoptions(linewidth=1000)
tr.set_printoptions(linewidth=1000)
tr.set_default_dtype(tr.float64)

# helper to kron sequence of matrices
def kron(*args):
    if len(args) == 1: return args[0]
    return tr.kron(args[0], kron(*args[1:]))

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

# number of qubits
n_qb = 6

# form all of the kroneker products
pauli_prods = tr.zeros(9, 2**n_qb, 2**n_qb, dtype=tr.complex128)
p = 0
for a, b in it.product(range(4), range(1,4)):
    if (a,b) in ((0,1), (0,2), (1,1)): continue
    idx = tr.tensor((a, b) + (0,)*(n_qb - 2))
    for i in range(n_qb):
        terms = pauli[tr.roll(idx, i)]
        pauli_prods[p] += kron(*terms)
    p += 1

# print(pauli_prods)
# input('.')

def get_spectrum(c):
    # c is (K, 9) tensor, K is batch size
    H = (pauli_prods[None,:,:,:] * c[:,:,None,None]).sum(dim=1) # (K, 2**n_qb, 2**n_qb)
    spectrum = tr.linalg.eigvalsh(H) # (K, 2**n_qb)
    return spectrum

def get_loss_factory(c0):
    target_spectrum = get_spectrum(c0)

    def get_loss(c):
        # c is (K, 9) tensor, K is batch size
        spectrum = get_spectrum(c)
        return tr.sum((spectrum - target_spectrum)**2, dim=1) # (K,)

    return get_loss

def f_factory(get_loss):

    def f(v):
        # v is (9, K) numpy array, return (9, K) batch of gradients
        c = tr.tensor(v.T, requires_grad=True) # torch wants batch first
        loss = get_loss(c)
        loss.sum().backward()
        return c.grad.numpy().T

    return f

def Df_factory(get_loss):

    def Df(v):
        # v is (9, K) numpy array, return (K, 9, 9) batch of hessians (jacobian of gradients)
        hess_fun = tr.func.hessian(get_loss)
        Dfv = []
        for k in range(v.shape[1]):
            c = tr.tensor(v[:,k:k+1].T)
            Dfv.append(hess_fun(c).squeeze()) # squeeze singleton batch dimensions
        Dfv = tr.stack(Dfv)
        return Dfv.numpy()

    return Df

def ef(v): return 1e-9

if __name__ == "__main__":

    mp.rcParams['font.family'] = 'serif'
    mp.rcParams['text.usetex'] = True

    do_fiber = False

    if do_fiber:

        # random sample for target spectrum
        # c_targ = tr.randn(1,9)
        c_targ = tr.tensor([[-.5, -.15, 1.1, .33, .75, .3, .3, -.8, -.75]]) # two optima
        get_loss = get_loss_factory(c_targ)
        f = f_factory(get_loss)
        Df = Df_factory(get_loss)

        # get initial fiber point
        v0 = c_targ.numpy().T

        # # f(v0) = 0, so use default random choice of direction vector
        # c_dir = None

        # constant direction that finds two optima
        c_dir = np.array([[-0.08657043, 0.10342339, 0.22536035, 0.39073649, 0.01098348,-0.44329002, 0.1645999,-0.08833236,-0.21927355]]).T

        # Set up fiber arguments
        fiber_kwargs = {
            "f": f,
            "Df": Df,
            "ef": ef,
            "compute_step_amount": lambda trace: (0.0001, 0, False),
            "v": v0,
            "c": c_dir,
            "terminate": lambda trace: (get_loss(trace.x[:9].T) > 0.001).any(),
            "max_step_size": 1,
            "max_traverse_steps": 5000,#000,
            "max_solve_iterations": 2**5,
            "logger": Logger(sys.stdout),
        }

        # Run in one direction
        solution = sv.fiber_solver(**fiber_kwargs)
        X1 = np.concatenate(solution["Fiber trace"].points, axis=1)
        V1 = X1[:-1,:]
        A1 = X1[-1,:]
        R1 = solution["Fixed points"]
        z = solution["Fiber trace"].z_initial
        print(len(A1))

        # Run in other direction (negate initial tangent)
        fiber_kwargs["z"] = -z
        solution = sv.fiber_solver(**fiber_kwargs)
        X2 = np.concatenate(solution["Fiber trace"].points, axis=1)
        V2 = X2[:-1,:]
        A2 = X2[-1,:]
        R2 = solution["Fixed points"]
        print(len(A2))

        # Join fiber segments and roots
        V = np.concatenate((np.fliplr(V1), V2), axis=1)
        A = np.concatenate((A1[::-1], A2), axis=0)
        R = np.concatenate((R1, R2), axis=1)

        R, fixed = fx.refine_points(R, f, ef, Df)
        R = R[:, fixed]

        duplicates = lambda U, v: (np.fabs(U - v) < 1e-3).all(axis=0)
        R = fx.get_unique_points(R, duplicates)

        with open("qubits.pkl","wb") as f: pk.dump((c_targ, V, A, R), f)

    with open("qubits.pkl","rb") as f: (c_targ, V, A, R) = pk.load(f)

    get_loss = get_loss_factory(c_targ)
    f = f_factory(get_loss)

    C = f(V)
    print("constant direction vectors:")
    print(C[:,:3])

    # # filter duplicates
    # duplicates = lambda U, v: (np.fabs(U - v) < 1e-3).all(axis=0)
    # R = fx.get_unique_points(R, duplicates)

    # filter stationary points with non-zero loss
    loss = get_loss(tr.tensor(R.T))
    keep = (loss.numpy() < 1e-7)
    R = R[:,keep]
    loss = get_loss(tr.tensor(R.T))

    print(f"{R.shape[1]} optima")

    targ_spectrum = get_spectrum(c_targ)
    spectrums = get_spectrum(tr.tensor(R.T))
    diffs = spectrums - targ_spectrum

    print(f"loss = {loss}")
    print("f(R)", np.fabs(f(R)).max(axis=0))
    print(R)

    trace_loss = get_loss(tr.tensor(V.T))
    local_min = np.flatnonzero((trace_loss[1:-1] <= trace_loss[2:]) & (trace_loss[1:-1] <= trace_loss[:-2]))

    # pt.plot(A)
    # pt.show()

    fig = pt.figure(figsize=(4,2))

    pt.subplot(1,2,1)
    # pt.plot(diffs.numpy().T)
    # pt.xlabel("$i$")
    # pt.ylabel("$\\lambda_i - \\lambda^*_i$")

    pt.plot(trace_loss, 'k-')
    pt.plot(local_min[0], 0, 'bo')
    pt.plot(local_min[1], 0, 'ro')
    pt.ylabel("$||\\Lambda - \\Lambda_0||^2$")
    pt.xlabel("Step along fiber")

    pt.subplot(1,2,2)
    pt.plot(R[:,0], 'b-')
    pt.plot(R[:,1], 'r-')
    pt.xlabel("Coefficient index")
    pt.ylabel("Coefficient value")
    pt.ylim([-1.1, 1.1])

    # fig.suptitle("SA2")
    pt.tight_layout()
    pt.show()

