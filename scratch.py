import numpy as np
import matplotlib.pyplot as plt
import dfibers.directional_fibers as df
import dfibers.solvers as sv
import dfibers.examples.rnn as rnn

N = 2
W = 1.25*np.eye(N) + 0.1*np.random.randn(N,N)
f = rnn.f_factory(W)
ef = rnn.ef_factory(W)
Df = rnn.Df_factory(W)
compute_step_amount = rnn.compute_step_amount_factory(W)
x = 0.01*np.random.randn(N+1,1)
c = np.random.randn(N,1)
c = c/np.linalg.norm(c)
terminate = rnn.terminate_factory(W, c)
max_solve_iterations = 2**5
solve_tolerance = 10**-18
max_step_size = 1

solution = sv.fiber_solver(
    f,
    ef,
    Df,
    compute_step_amount,
    v=x[:N,:],
    c=c,
    terminate=terminate,
    max_step_size=0.01,
    max_traverse_steps=1000,
    max_solve_iterations=max_solve_iterations,
    solve_tolerance=solve_tolerance,
    )
    
fiber = solution["Fiber"]
X = fiber["X"]
z = fiber["z"]
print("%d steps"%X.shape[1])
# X = np.concatenate((-np.fliplr(X), X), axis=1)
V = X[:-1,:]
C = f(V)
lm = 1.25
lm_idx = (np.fabs(V) < lm).all(axis=0)
# plt.subplot(1,2,1)
plt.plot(V[0,:],V[1,:],'b-')
plt.gca().quiver(V[0,lm_idx],V[1,lm_idx],C[0,lm_idx],C[1,lm_idx],scale=.005,units='dots',width=2,headwidth=5)
print("z:")
print(z)
plt.plot([V[0,0], V[0,0]+z[0,0]],[V[1,0],V[1,0]+z[1,0]],'m-')


for r in solution["Refinements"]:
    plt.plot(r["X"][0,0], r["X"][1,0], 'go')

for r in solution["Refinements"]:
    plt.plot(r["X"][0,:], r["X"][1,:], 'g.-')

V = solution["Fixed points"]
plt.plot(V[0,:],V[1,:],'ro')

plt.xlim((-lm,lm))
plt.ylim((-lm,lm))

# plt.subplot(1,2,2)
# steps = fiber["steps"].cumsum()
# a = X[-1,:-1]
# plt.plot(steps, a, 'b-')
# fixed_index = solution["Fixed index"][:-1]
# plt.plot(steps[fixed_index], a[fixed_index], 'ro')
# last = fixed_index[-2]
# plt.xlim((0,steps[last]))
# plt.ylim((a[:last].min(),a[:last].max()))

plt.show()
