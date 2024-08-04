"""
Example RNN using softplus (smooth relaxation of relu)
Like pytorch, softplus will revert to linear function for inputs above 20 for numerical stability
"""
import numpy as np
import matplotlib.pyplot as plt
import dfibers.traversal as tv
import dfibers.solvers as sv
import dfibers.fixed_points as fx

# Define the dynamical system
N = 2
# W = np.random.randn(N,N)
# W = 0.01*np.random.rand(N,N)
# W = 1./np.random.rand(N,N)
W = np.array([[2., 0], [0, 3]])
b = np.array([[-2., -3.]]).T
def f(v):
    # softplus input
    x = W @ v + b
    # softplus output
    y = np.where(x > 20., x, np.log(1. + np.exp(x)))
    # update vector
    return y - v

# define Jacobian
I = np.eye(W.shape[0])
def Df(V):
    # softplus derivative
    x = W @ V + b
    D = np.where(x > 20, 1., np.exp(x) / (1. + np.exp(x)))
    # use newaxis to implement left-multiply with diagonal matrix
    return D.T[:,:,np.newaxis]*W[np.newaxis,:,:] - I[np.newaxis,:,:]

# define "forward error" (use constant tolerance for simplicity)
ef = lambda v: 10**-10

# define termination criteria (when state gets large)
terminate = lambda trace: (np.fabs(trace.x) > 10**6).any()

# define adaptive step size computation (constant here for simplicity)
compute_step_amount = lambda trace: (10**-2, None, False)

# run the solver
# v0 = np.random.randn(N,1)
v0 = np.array([[1.,-0.3]]).T # fiber containing all fixed points
solution = sv.fiber_solver(
    f,
    ef,
    Df,
    compute_step_amount,
    v = v0,
    c = f(v0),
    terminate=terminate,
    max_traverse_steps=1000,
    max_solve_iterations=2**5,
)

# remove duplicates and sanitize fixed points
duplicates = lambda U, v: (np.fabs(U - v) < 2**-21).all(axis=0)
V = solution["Fixed points"]
fxpts = fx.sanitize_points(V, f, ef, Df, duplicates)

# Extract steps along fiber and corresponding f(v)'s
trace = solution["Fiber trace"]
X = np.concatenate(trace.points, axis=1)
V = X[:-1,:]
V = V[:, ::10] # downsample fiber for fewer quiver arrows

print("v0, c:")
print(v0)
print(f(v0))

print("trace:")
print(V)
print("Fixed points:")
print(fxpts)
print("Fixed point residuals:")
print(f(fxpts))
assert((f(fxpts) < ef(fxpts)).all())

# Plot fiber and fixed points
x_min, x_max = V[0].min()-.5, V[0].max()+.5
y_min, y_max = V[1].min()-.5, V[1].max()+.5
X_grid, Y_grid = np.mgrid[x_min:x_max:20j,y_min:y_max:20j]
plt.figure(figsize=(5,4.5))
tv.plot_fiber(X_grid, Y_grid, V, f, scale_XY=None, scale_V=None)
plt.scatter(*fxpts, color='k', marker='o')
plt.xlabel("v1")
plt.ylabel("v2",rotation=0)
plt.tight_layout()
plt.savefig("rnn_softplus.png")
plt.show()

