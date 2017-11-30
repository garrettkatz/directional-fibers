import numpy as np
import matplotlib.pyplot as plt
import directional_fibers as df
import examples.rnn as rnn

N = 2
W = 1.25*np.eye(N) + 0.1*np.random.randn(N,N)
f = rnn.f_factory(W)
Df = rnn.Df_factory(W)
compute_step_size = rnn.compute_step_size_factory(W)
x = 0.01*np.random.randn(N+1,1)
c = np.random.randn(N,1)
c = c/np.linalg.norm(c)
max_solve_iterations = 2**5
solve_tolerance = 10**-18
max_step_size = 1


result = df.traverse_fiber(
    f,
    Df,
    compute_step_size,
    v=x[:N,:],
    c=c,
    max_traverse_steps=1000,
    max_solve_iterations=max_solve_iterations,
    solve_tolerance=solve_tolerance,
    )
X = np.concatenate(result["X"], axis=1)
X = np.concatenate((-np.fliplr(X), X), axis=1)
V = X[:-1,:]
lm = 1.25
V = V[:, (np.fabs(V) < lm).all(axis=0)]
C = f(V)
plt.plot(V[0,:],V[1,:],'b.')
plt.gca().quiver(V[0,:],V[1,:],C[0,:],C[1,:],scale=.005,units='dots',width=2,headwidth=5)
plt.xlim((-lm,lm))
plt.ylim((-lm,lm))
plt.show()
