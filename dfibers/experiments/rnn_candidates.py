"""
Measures improvement on RNN fixed point location of local minima of |alpha| and within-fiber refinement
"""
import sys
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import dfibers.logging_utilities as lu
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.examples.rnn as rnn

def run_experiment(result_filename, network_sampling):

    results = {}
    
    # logger = lu.Logger(sys.stdout)
    logger = lu.Logger(open("tmp.log","w"))
    for (N,S) in network_sampling.items():
        for sample in range(S):
        
            # Sample network
            f, Df, ef, W, V = rnn.make_known_fixed_points(N)
    
            # Run fiber solver
            fxpts, solution = rnn.run_fiber_solver(W,
                max_traverse_steps = 2**14,
                logger=logger.plus_prefix("(%d,%d): "%(N,sample)),
                abs_alpha_min = True,
                within_fiber = True)
    
            # Get pre-refinement sign change candidates
            sign_changes = solution["Sign changes"]
            refinements = solution["Refinements"]
            candidates = np.concatenate( # first refinement fiber point
                [refinements[sc].points[0][:-1,:] for sc in sign_changes],
                axis = 1)
    
            # Post-process sign change candidates
            logger.log("(%d,%d): Sanitizing %d sc points"%(
                N,sample, 2*candidates.shape[1] + 1))
            sign_change_fxpts = fx.sanitize_points(
                np.concatenate(
                    (-candidates, np.zeros((N,1)), candidates),
                    axis=1),
                f, ef, Df,
                duplicates = rnn.duplicates_factory(W),
            )
    
            # Count union
            logger.log("(%d,%d): Sanitizing %d+%d=%d union points"%(
                N,sample, fxpts.shape[1], sign_change_fxpts.shape[1],
                fxpts.shape[1] + sign_change_fxpts.shape[1]))
            union_fxpts = fx.sanitize_points(
                np.concatenate(
                    (fxpts, sign_change_fxpts),
                    axis=1),
                f, ef, Df,
                duplicates = rnn.duplicates_factory(W),
            )
    
            print("(%d,%d) fxpt counts: %d, %d < %d"%(
                N,sample,fxpts.shape[1], sign_change_fxpts.shape[1],
                union_fxpts.shape[1]))
                
            results[(N,sample)] = {
                "new count": fxpts.shape[1],
                "old count":  sign_change_fxpts.shape[1],
                "union": union_fxpts.shape[1],
                }

            with open(results_filename,'w') as f: pk.dump(results, f)

def plot_results(results_filename, network_sampling):
    network_sizes = network_sampling.keys()
    network_sizes.sort()
    colors = np.linspace(.75, 0., len(network_sizes)) # light to dark
    
    with open(results_filename,'r') as f: results = pk.load(f)
    for i, N in enumerate(network_sizes):
        S = network_sampling[N]
        old_counts = [results[N,s]["old count"] for s in range(S)]
        new_counts = [results[N,s]["new count"] for s in range(S)]
        rgba = [colors[i]]*3 + [1]
        pt.plot(old_counts, new_counts, 'o', fillstyle='none', markeredgecolor = rgba)
    pt.legend(network_sizes)
    pt.xlabel("Old counts")
    pt.xlabel("New counts")
    pt.show()
    
if __name__ == "__main__":

    results_filename = "rnn_candidates_results.pkl"

    # Maps network size: sample size
    network_sampling = {
        3: 3,
        4: 3,
        9: 3,
        # 27: 10,
        # 81: 10,
        # 243: 10,
    }

    run_experiment(results_filename, network_sampling)
    plot_results(results_filename, network_sampling)
