# directional-fibers

`directional-fibers` is a fixed point solver for discrete or continuous time dynamical systems with states in multi-dimensional Euclidean space.  The solver works by numerically traversing directional fibers, which are curves in high-dimensional state space that contain fixed points:

![Directional Fiber 1](https://cloud.githubusercontent.com/assets/6537102/21059296/bc76324e-be0f-11e6-9b5f-24a3cc928711.png)

More information is available in this publication:

[Katz, G. E., Reggia, J. A. (2017). Using Directional Fibers to Locate Fixed Points of Recurrent Neural Networks. IEEE Transactions on Neural Networks and Learning Systems (accepted). IEEE.](https://doi.org/10.1109/TNNLS.2017.2733544) (Here is a [preprint](https://www.cs.umd.edu/~gkatz/TNNLS-2016-P-7293.R2.pdf)).

## Requirements

`directional-fibers` has been tested using the following environment, but it may work with other operating systems and versions.
* [Fedora](https://getfedora.org/) 26
* [Python](https://www.python.org/) 2.7.14
* [numpy](http://www.numpy.org/) 1.14.0
* [scipy](http://www.scipy.org/scipylib/index.html) 1.0.0
* [matplotlib](http://matplotlib.org/) 2.0.0

## Installation

1. [Clone or download](https://help.github.com/articles/cloning-a-repository/) this repository into a directory of your choice.
2. Add the local repository directory to your [PYTHONPATH](https://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH).

## Basic Usage

To find fixed points with `directional-fibers`, first you need to define your dynamical system.  Only dynamical systems with states in Euclidean space can be used.  Points in N-dimensional Euclidean space are represented by Nx1 numpy arrays.  You must define a function `f` that defines the change in the system over time - either a discrete time difference or a continuous-time derivative.  `f` must take an NxK numpy array as input and return an NxK numpy array as output, where the k^th column of the output is `f` applied to the k^th column of the input.  For example:

```python
>>> import numpy as np
>>> N = 2
>>> W = np.random.randn(N,N)
>>> f = lambda v: np.tanh(W.dot(v)) - v
```

Next you must define a function `Df` that computes the NxN Jacobian of `f`.  Only dynamical systems with differentiable `f` can be used.  If v has more than one column, then Df should be a 3d np.array, where `Df(v)[k,:,:]` is the Jacobian evaluated at `v[:,[k]]`.  Continuing the example:

```python
>>> I = np.eye(W.shape[0])
>>> def Df(V):
...     D = 1-np.tanh(W.dot(V))**2
...     if V.shape[1] == 1: return D*W - I
...     else: return D.T[:,:,np.newaxis]*W[np.newaxis,:,:] - I[np.newaxis,:,:]
...
```

Next you must define a function `ef` that bounds the forward error in `f`: the difference between the true mathematical value of `f` and its finite-precision machine approximation.  A point `v` is considered fixed when `(np.fabs(f(v)) < ef(v)).all()`. `ef` is also used during directional fiber traversal to keep residual errors near machine precision.  Since `ef` essentially plays the role of a tolerance, as a simpler alternative you can have it return a constant value:

```python
>>> ef = lambda v: 10**-6
```

As directional fibers are traversed, the current and past traversal data (points along the fiber, residual errors, etc.) are saved in a `FiberTrace` object:

```python
>>> import dfibers.traversal as tv
>>> help(tv.FiberTrace)
```

You can define a custom termination criterion that takes a `FiberTrace` object as input, and returns `True` when the criterion is satisfied at the current step.  For example, termination when the system state gets very large:

```python
>>> terminate = lambda trace: (np.fabs(trace.x) > 10**6).any()
```

Alternatively there are also standard termination criteria such as maximum run time or maximum number of steps that can be specified later as key-word arguments.  In that case you can set `terminate = None`.

Lastly, you should define a `compute_step_amount` function, which computes a reasonable step size for the current numerical step along the fiber.  This function should take a `FiberTrace` object as input, and return two outputs: `step_amount`, the actual size of the step, and `step_data`, which can contain any additional step-related data of interest and will be saved for your post-traversal analysis.  In the simplest case, you can return a small constant step size and no additional data:

```python
>>> compute_step_amount = lambda trace: (10**-3, None)
```

Finally, you can now run the solver:

```python
>>> import dfibers.solvers as sv
>>> help(tv.traverse_fiber)
>>> help(sv.fiber_solver)
>>> solution = sv.fiber_solver(
... f,
... ef,
... Df,
... compute_step_amount,
... N=N,
... terminate=terminate,
... max_traverse_steps=10**3,
... )
```

Candidate fixed points can be post-processed by removing duplicates and points that are not quite fixed.  For this you must define a `duplicates` function that determines when two points should be considered duplicates.  `duplicates` should take two inputs: an NxK matrix `V` and Nx1 vector `v`.  It should return one output, a length N boolean array whose k^th element is `True` if `V[:,[k]] and `v` should be considered duplicates:

```python
>>> import dfibers.fixed_points as fx
>>> duplicates = lambda V, v: (np.fabs(V - v) < 2**-21).all(axis=0)
>>> V = solution["Fixed points"]
>>> V = fx.sanitize_points(V, f, ef, Df, duplicates)
```

Inspect the solutions:

```python
>>> print("Fixed points:")
>>> print(V)
>>> print("Fixed point residuals:")
>>> print(f(V))
>>> assert((f(V) < ef(V)).all())
```

More sophisticated examples can be found in the `directional-fibers/dfibers/examples` sub-directory.
