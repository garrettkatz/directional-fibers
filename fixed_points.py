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

