"""
Probes existence of a single c that locates all fixed points of RNN
"""
import sys, time
import pickle as pk
import itertools as it
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
            dups = rnn.duplicates_factory(W)
    
            # Keep fiber-solving until a c finds all V
            for trial in it.count(0):
                print(N,S,trial)

                # Run fiber solver
                start_time = time.clock()
                fxpts, solution = rnn.run_fiber_solver(W,
                    max_traverse_steps = 2**15,
                    logger=logger.plus_prefix("(%d,%d,%d): "%(N,sample,trial)))
                stop_time = time.clock()

                # Update union of fxpts
                V = fx.sanitize_points(
                    np.concatenate((fxpts, V), axis=1),
                    f, ef, Df,
                    duplicates = dups,
                )

                # Save results
                trace = solution["Fiber trace"]
                results[N,S,trial] = {
                    "status": trace.status,
                    "c": trace.c,
                    "seconds": stop_time - start_time,
                    "|fxpts|": fxpts.shape[1],
                    "|V|": V.shape[1],
                    }
                with open(results_filename,'w') as rf: pk.dump(results, rf)

                print("%d,%d,%d: status %s, |fxpts|=%d, |V|=%d"%(
                    N,sample,trial,trace.status,fxpts.shape[1],V.shape[1]))

                # Done if current c found full union
                if fxpts.shape[1] == V.shape[1]: break

            print("%d,%d: %d trials"%(N,sample,trial+1))

if __name__ == "__main__":

    results_filename = "rnn_c_results.pkl"

    # Maps network size: sample size
    network_sampling = {
        5: 1,
        # 3: 50,
        # 4: 50,
        # 9: 50,
        # 27: 10,
        # 81: 10,
        # 243: 10,
    }

    run_experiment(results_filename, network_sampling)
    # plot_results(results_filename, network_sampling)
