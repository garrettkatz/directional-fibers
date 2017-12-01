import numpy as np
import directional_fibers as df

def is_fixed(f, ef, v):
    """
    Decide whether a point is fixed.
    f: dynamical difference function
    ef: estimate of f forward-error
    v: each column is a point to check.
    returns (fixed, error) where
        fixed is False if v is definitely not fixed
        error is ef(v)
    """
    error = ef(v)
    fixed = (np.fabs(f(v)) < error).all(axis=0)
    return fixed, error

def get_connected_components(V, E):
    """
    Find all connected components in an undirected unweighted graph.
    V should be a 2D numpy.array, where V[:,p] is associated with the p^{th} node.
    E should be an edge indicator function, i.e.
      E(V, u)[p] == True iff there is an edge between V[:,p] and u.
    Returns components, a 1D numpy.array where
      components[p] == components[q] iff V[:,p] and V[:,q] are in the same connected component.
    """
    # Initialize each point in isolated component
    components = np.arange(V.shape[1])
    # Merge components one point at a time
    for p in range(V.shape[1]):
        # Index neighbors to current point
        neighbors = E(V[:,:p+1], V[:,[p]])
        # Merge components containing neighbors
        components[:p+1][neighbors] = components[:p+1][neighbors].min()
    return components

def get_unique_points(V, duplicates, base=2):
    """
    Extract unique points from a set with potential duplicates.
    V should be a 2D numpy.array, where V[:,p] is the p^{th} point.
    duplicates should be a function handle, where
      duplicates(V, v)[p] == True iff V[:,p] and v are to be considered duplicate points.
    Recursively divides and conquers V until V.shape[1] <= base.
    Returns U, where U[:,q] is the q^{th} extracted unique point.
    """
    if V.shape[1] > base:
        # Divide and conquer
        split_index = int(V.shape[1]/2)
        U1 = get_unique_points(V[:,:split_index], duplicates, base=base)
        U2 = get_unique_points(V[:,split_index:], duplicates, base=base)
        # Prepare to merge result
        V = np.concatenate((U1, U2),axis=1)
    # Get connected components of neighbor graph
    components = get_connected_components(V, duplicates)
    # Extract representatives from each component
    _, representative_index = np.unique(components, return_index=True)
    U = V[:,representative_index]
    return U

def fiber_solver(
    f,
    ef,
    Df,
    compute_step_amount,
    v=None,
    c=None,
    N=None,
    terminate=None,
    logfile=None,
    stop_time=None,
    max_traverse_steps=None,
    max_step_size=None,
    max_solve_iterations=None,
    solve_tolerance=None,
    ):
    """
    Fixed point location using directional fibers.
    All parameters are as described in directional_fibers.traverse_fiber().
    Returns solution, a dictionary with keys
        "Fiber": the result of traverse_fiber()
        "Fixed points": an array with one fixed point per column
        "Refinements": a list of local traverse_fiber results, one per fixed point
        "Fixed index": an array of fixed point candidate indices in the fiber
    """

    # Traverse fiber
    fiber_result = df.traverse_fiber(
        f,
        Df,
        compute_step_amount,
        v=v,
        c=c,
        N=N,
        terminate=terminate,
        logfile=logfile,
        stop_time=stop_time,
        max_traverse_steps=max_traverse_steps,
        max_step_size=max_step_size,
        max_solve_iterations=max_solve_iterations,
        solve_tolerance=solve_tolerance,
    )
    
    # Keep final direction vector in case random default was used (in theory shouldn't matter)
    c = fiber_result["c"]

    # Extract candidate fixed points: endpoints, local |alpha| minimum, alpha sign change
    # As well as immediate neighbors for close pairs of fixed points and added redundancy
    X = fiber_result["X"]
    a = X[-1,:]
    fixed_index = np.zeros(len(a), dtype=bool)
    # endpoints
    fixed_index[[0, -1]] = True
    # local magnitude minima
    fixed_index[1:-1] |= (np.fabs(a[1:-1]) <= np.fabs(a[2:])) & (np.fabs(a[1:-1]) <= np.fabs(a[:-2]))
    # sign changes
    fixed_index[:-1] |= np.sign(a[:-1]) != np.sign(a[1:])
    # extra redundancy with neighbors
    fixed_index[:-1] = np.logical_or(fixed_index[:-1], fixed_index[1:])
    fixed_index[1:] = np.logical_or(fixed_index[1:], fixed_index[:-1])

    # Set up within-fiber Newton-Raphson step computation
    def compute_refine_step_amount(x, DF, z):
        refine_step_amount = -x[-1,0]/z[-1,0]
        fiber_step_amount, fiber_step_data = compute_step_amount(x, DF, z)
        step_amount = np.sign(refine_step_amount)*min(np.fabs(refine_step_amount), fiber_step_amount)
        step_data = (refine_step_amount, fiber_step_amount, fiber_step_data)
        return step_amount, step_data
    refine_terminate = lambda x: is_fixed(f, ef, x[:-1,:])[0] or terminate(x)

    # Run within-fiber Newton-Raphson at each candidate
    X = X[:, fixed_index]
    refinement_results = []
    fixed_points = []
    for i in range(X.shape[1]):
        refinement_result = df.traverse_fiber(
            f,
            Df,
            compute_refine_step_amount,
            v=X[:-1,[i]].copy(),
            c=c,
            terminate=refine_terminate,
            logfile=logfile,
            stop_time=stop_time,
            max_traverse_steps=2**5, # only traverse enough for Newton-Raphson solve
            max_step_size=max_step_size,
            max_solve_iterations=max_solve_iterations,
            solve_tolerance=solve_tolerance,
        )
        refinement_results.append(refinement_result)
        fixed_points.append(refinement_result["X"][:-1,[-1]].copy())
        # fixed_points.append(refinement_result["X"][:-1,[0]].copy())
    
    # Return output
    solution = {
        "Fiber": fiber_result,
        "Fixed points": np.concatenate(fixed_points,axis=1),
        "Refinements": refinement_results,
        "Fixed index": np.flatnonzero(fixed_index),
    }
    return solution
        
