import numpy as np
import numerical_utilities as nu

def is_fixed(V, f, ef):
    """
    Decide whether points are fixed.
    V: (N,P) ndarray of points to check, one per column
    f: dynamical difference function
    ef: estimate of f forward-error
    returns (fixed, error) where
        fixed[p] is False if V[:,p] is definitely not fixed
        error = ef(V)
    """
    error = ef(V)
    fixed = (np.fabs(f(V)) < error).all(axis=0)
    return fixed, error

def refine_points(V, f, ef, Df, max_iters=2**5, batch_size=100):
    """
    Refine candidate fixed points with Newton-Raphson iterations
    V: (N,P)-ndarray, one candidate point per column
    f, ef: as in is_fixed
    Df(V)[p,:,:]: derivative of f(V[:,p])
    max_iters: maximum Newton-Raphson iterations performed
    batch_size: at most this many points solved for at a time (limits memory usage)
    returns (V, fixed) where
        V[:,p] is the p^th point after refinement
        fixed[p] is True if the p^th point is fixed
    """

    # Split points into batches
    num_splits = int(np.ceil(float(V.shape[1])/batch_size))
    point_batches = np.array_split(V, num_splits, axis=1)
    fixed_batches = []

    for b in range(num_splits):

        # Refine current batch with Newton-Raphson
        X, converged, _, _ = nu.nr_solves(
            point_batches[b], f, Df, ef, max_iterations=max_iters)
        point_batches[b] = X
        fixed_batches.append(converged)

    # Format output
    return np.concatenate(point_batches, axis=1), np.concatenate(fixed_batches)

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

def sanitize_points(V, f, ef, Df, duplicates, base=2):
    """
    Sanitize a set of candidate fixed points
    Refines and removes non-fixed points and duplicates
    V: (N,P) ndarray of P candidate points, one per column
    f, ef, Df: as in refine_points
    duplicates, base: as in get_unique_points
    Returns (N,K) ndarray of K unique fixed points
    """
    V, fixed = refine_points(V, f, ef, Df)
    return get_unique_points(V[:,fixed], duplicates, base=base)
