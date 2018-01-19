import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
print(__name__)
import os
print(os.path.abspath(__file__))
# __package__ = "dfibs"
print(__package__)
# from .. import solvers as sv

xy = np.mgrid[-3:3:30j, -3:3:30j]
x, y = xy[0], xy[1]
z = .1*x**4 + .1*y**4 - .75*(x+.1)**2 - .75*(y+.2)**2
# z = x*y + x**2
# dx = y + 2x, dy = x
dx = .1*4*x**3 - .75*2*(x+.1)
dy = .1*4*y**3 - .75*2*(y+.2)

N = 2
f = lambda v: .1*4*v**3 - .75*2*(v + np.array([[.1],[.2]]))
ef = lambda v: .001*np.ones(v.shape)
Df = lambda v: np.diagflat(.1*4*3*v**2 - .75*2)
compute_step_amount = lambda x, DF, z: (0.01, 0)
v = np.zeros((N,1))
c = f(v)
c = c/np.linalg.norm(c)

solution = sv.fiber_solver(
    f,
    ef,
    Df,
    compute_step_amount,
    v=v,
    c=c,
    max_step_size=1,
    max_traverse_steps=2000,
    max_solve_iterations = 2**4,
    )
X = solution["Fiber"]["X"]
V = X[:-1,::25]
C = f(V)
lm = x.max()
lm_idx = (np.fabs(V) < lm).all(axis=0)

plt.subplot(1,2,1)
plt.contour(x, y, z, 20)
# plt.subplot(1,3,2)
# plt.quiver(x, y, dx, dy,scale=.005,units='dots',width=2,headwidth=5)
# plt.subplot(1,2,1)
plt.plot(V[0,:],V[1,:],'b-')
plt.gca().quiver(V[0,lm_idx],V[1,lm_idx],C[0,lm_idx],C[1,lm_idx],scale=.05,units='dots',width=2,headwidth=5)


ax = plt.gcf().add_subplot(1,2,2,projection="3d")
ax.plot_surface(x,y,z)
plt.show()
