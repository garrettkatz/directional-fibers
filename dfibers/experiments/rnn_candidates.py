"""
Measures improvement on RNN fixed point location of local minima of |alpha| and within-fiber refinement
"""
import numpy as np
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.examples.rnn as rnn

# Maps network size: sample size
network_sampling = {
    # 3: 10,
    10: 10,
}

for (N,S) in network_sampling.items():
    for sample in range(S):
    
        # Sample network
        f, Df, ef, W, V = rnn.make_known_fixed_points(N)

        # Run fiber solver
        fxpts, solution = rnn.run_fiber_solver(W,
            max_traverse_steps = 10000,
            abs_alpha_min = True,
            within_fiber = True)

        # Get pre-refinement sign change candidates
        sign_changes = solution["Sign changes"]
        refinements = solution["Refinements"]
        candidates = np.concatenate( # first refinement fiber point
            [refinements[sc].points[0][:-1,:] for sc in sign_changes],
            axis = 1)

        # Post-process sign change candidates
        sign_change_fxpts = fx.sanitize_points(
            np.concatenate(
                (-candidates, np.zeros((N,1)), candidates),
                axis=1),
            f, ef, Df,
            duplicates = rnn.duplicates_factory(W),
        )

        print("(%d,%d) fxpt counts: %d, %d"%(
            N,sample,fxpts.shape[1], sign_change_fxpts.shape[1]))
