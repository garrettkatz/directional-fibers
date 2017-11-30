import numpy as np
import directional_fibers as df

def fiber_solver(
    f,
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
            terminate=terminate,
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
        
