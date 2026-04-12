"""
Fiber-based fixed point location in the 2-variable quadratic system
    f([x,y].T) = M @ [1, x, y, xy, x**2, y**2].T
"""

import numpy as np
import matplotlib.pyplot as pt
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

def f_factory(M):
    """
    For a given 2x6 coefficient matrix M, returns the function f,
    where f(V)[:,p] = f(V[:,p])
    """
    M = M.copy() # save in closure in case modified later
    def f(V):
        x, y = V
        Q = np.stack([np.ones(len(x)), x, y, x*y, x**2, y**2])
        return M @ Q
    return f
    
def Df_factory(M):
    """
    For a given 2x6 coefficient matrix M, returns the function Df,
    where Df(V)[p,:,:] is the Jacobian of f at V[:,[p]]
    """
    def Df(V):
        x, y = V
        dQ = np.zeros((len(x), 6, 2))
        dQ[:,1,0] = 1
        dQ[:,3,0] = y
        dQ[:,4,0] = 2*x
        dQ[:,2,1] = 1
        dQ[:,3,1] = x
        dQ[:,5,1] = 2*y
        return M @ dQ
    return Df

def ef(v):
    """
    Simple forward error bound (TODO)
    """
    return 0.001*np.ones(v.shape) # placeholder

def plot_conic_sections(M, X, Y):
    # use countour with X, Y as meshgrid (based on https://mmas.github.io/conics-matplotlib)
    f = f_factory(M)
    V = np.stack([X.flatten(), Y.flatten()])
    F = f(V)
    pt.contour(X, Y, F[0].reshape(X.shape), levels=[0], colors='blue')
    pt.contour(X, Y, F[1].reshape(X.shape), levels=[0], colors='red')

def reference_roots(M):
    """
    For a given 2x6 coefficient matrix M, return the real roots of the system
    Eliminates y to get quartic in x
    """

    # save function for check before modifying M
    f = f_factory(M)

    # [TODO] special case: y^2 coefficients zero (not a major concern when M sampled randomly)
    if 0 in M[:,-1]:
        raise NotImplementedError

    # divide through y^2 coefficients
    M = M / M[:,-1:]

    # cancel y^2 terms
    M[0] = M[0] - M[1]

    # numerically solve quartic in x
    coefs = np.array([
        M[0,4]**2 - M[1,3]*M[0,3]*M[0,4] + M[0,3]**2*M[1,4],
        2*M[0,1]*M[0,4] - M[1,2]*M[0,3]*M[0,4] - M[1,3]*M[0,2]*M[0,4] - M[1,3]*M[0,3]*M[0,1] + 2*M[0,2]*M[0,3]*M[1,4] + M[0,3]**2*M[1,1],
        M[0,1]**2 + 2*M[0,0]*M[0,4] - M[1,2]*M[0,2]*M[0,4] - M[1,2]*M[0,3]*M[0,1] - M[1,3]*M[0,2]*M[0,1] - M[1,3]*M[0,3]*M[0,0] + M[0,2]**2*M[1,4] + 2*M[0,2]*M[0,3]*M[1,1] + M[0,3]**2*M[1,0],
        2*M[0,0]*M[0,1] - M[1,2]*M[0,2]*M[0,1] - M[1,2]*M[0,3]*M[0,0] - M[1,3]*M[0,2]*M[0,0] + M[0,2]**2*M[1,1] + 2*M[0,2]*M[0,3]*M[1,0],
        M[0,0]**2 - M[1,2]*M[0,2]*M[0,0] + M[0,2]**2*M[1,0],
    ])
    x = np.roots(coefs)

    # filter imaginary solutions
    real = np.fabs(np.imag(x)) < 1e-7
    if not real.any(): return [], []
    x = x[real].real

    # recover y
    y = -(M[0,0] + M[0,1]*x + M[0,4]*x**2) / (M[0,2] + M[0,3]*x)

    # checks
    powers = np.stack([x**4, x**3, x**2, x**1, x**0])
    resid = (coefs @ powers)
    assert (np.fabs(resid) < 1e-7).all()

    V = np.stack([x,y])
    fV = f(V)
    assert (np.fabs(fV) < 1e-7).all()

    return x, y
    

if __name__ == "__main__":

    # random quadric
    M = np.random.randn(2,6)
    f = f_factory(M)
    Df = Df_factory(M)

    # Lipschitz constant
    mu = 2 * np.linalg.norm(M[:,3:], ord=2)
    print(f"{mu=}")

    # Set up fiber arguments
    # v = np.array([[-.5],[-1.5]])
    # v = np.ones((2,1))
    v = np.random.randn(2,1)
    c = f(v)
    # v = None
    # c = np.array([[1],[.25]])
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        # "compute_step_amount": lambda trace: (0.01, 0, False),
        "compute_step_amount": tv.compute_lipschitz_step_amount_factory(mu),
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:2,:]) > 10).any(),
        "max_step_size": 1,
        "max_traverse_steps": 10000,
        "max_solve_iterations": 2**5,
    }

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,:]
    z = solution["Fiber trace"].z_initial
    
    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:-1,:]

    # Join fiber segments
    V = np.concatenate((np.fliplr(V1), V2), axis=1)
    V = V[:,np.isfinite(V).all(axis=0)]
    V = V[:,(np.fabs(V) < 3).all(axis=0)]

    # Get ground truth roots for validation
    xr, yr = reference_roots(M)    

    # Grid extents
    xlo, xhi = V[0].min(), V[0].max()
    ylo, yhi = V[1].min(), V[1].max()
    if len(xr) > 0:
        xlo = min(xlo, xr.min())
        xhi = max(xhi, xr.max())
        ylo = min(ylo, yr.min())
        yhi = max(yhi, yr.max())

    # Grids for fiber and attractor
    X_fiber, Y_fiber = np.mgrid[xlo-1:xhi+1:50j, ylo-1:yhi+1:50j]

    # Visualize fiber, conic sections, and roots
    pt.figure(figsize=(3.5,3.5))

    ax_fiber = pt.gca()
    # tv.plot_fiber(X_fiber, Y_fiber, V[:,::10], f, ax=ax_fiber, scale_XY=10, scale_V=10)
    tv.plot_fiber(X_fiber, Y_fiber, V, f, ax=ax_fiber, scale_XY=10, scale_V=10)
    pt.plot(V2[0,0], V2[1,0], 'ro')

    plot_conic_sections(M, X_fiber, Y_fiber)
    pt.plot(xr, yr, 'go')

    ax_fiber.set_xlabel("x")
    ax_fiber.set_ylabel("y",rotation=0)
    # pt.yticks(np.linspace(-2,2,5))
    pt.tight_layout()
    pt.show()

