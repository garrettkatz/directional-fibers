# Code from README.md

import numpy as np
N = 2
W = 1.2*np.eye(N) + 0.1*np.random.randn(N,N)
f = lambda v: np.tanh(W.dot(v)) - v

I = np.eye(W.shape[0])
def Df(V):
    D = 1-np.tanh(W.dot(V))**2
    if V.shape[1] == 1: return D*W - I
    else: return D.T[:,:,np.newaxis]*W[np.newaxis,:,:] - I[np.newaxis,:,:]

ef = lambda v: 10**-10

import dfibers.traversal as tv
# help(tv.FiberTrace)

terminate = lambda trace: (np.fabs(trace.x) > 10**6).any()

compute_step_amount = lambda trace: (10**-3, None)

import dfibers.solvers as sv
# help(tv.traverse_fiber)
# help(sv.fiber_solver)
solution = sv.fiber_solver(
    f,
    ef,
    Df,
    compute_step_amount,
    v = np.zeros((N,1)),
    terminate=terminate,
    max_traverse_steps=10**3,
    max_solve_iterations=2**5,
    )

import dfibers.fixed_points as fx
duplicates = lambda V, v: (np.fabs(V - v) < 2**-21).all(axis=0)
V = solution["Fixed points"]
V = fx.sanitize_points(V, f, ef, Df, duplicates)

print("Fixed points:")
print(V)
print("Fixed point residuals:")
print(f(V))
assert((f(V) < ef(V)).all())
