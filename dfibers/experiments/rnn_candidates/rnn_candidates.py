"""
Measures improvement on RNN fixed point location of local minima of |alpha| and within-fiber refinement
"""
import sys, os, time
import pickle as pk
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as pt
import dfibers.logging_utilities as lu
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.examples.rnn as rnn

def trialname(basename, N, sample):
    return "%s_N%d_s%d"%(basename, N, sample)

def run_trial(args):
    basename, N, sample, W, V, timeout = args

    logfile = open(trialname(basename,N,sample)+".log", "w")
    logger = lu.Logger(logfile)

    # Sample network
    f = rnn.f_factory(W)
    Df = rnn.Df_factory(W)
    ef = rnn.ef_factory(W)

    # Run fiber solver
    solve_logger = logger.plus_prefix("(%d,%d): "%(N,sample))
    stop_time = time.clock() + timeout
    fxpts, solution = rnn.run_fiber_solver(W,
        # max_traverse_steps = 2**15,
        stop_time = stop_time,
        logger=solve_logger,
        abs_alpha_min = True,
        within_fiber = True)
    status = solution["Fiber trace"].status

    # Get pre-refinement sign change candidates
    sign_changes = solution["Sign changes"]
    refinements = solution["Refinements"]
    candidates = np.concatenate( # first refinement fiber point
        [refinements[sc].points[0][:-1,:] for sc in sign_changes],
        axis = 1)

    # Post-process sign change candidates
    logger.log("(%d,%d): Sanitizing %d sc points\n"%(
        N,sample, 2*candidates.shape[1] + 1))
    sign_change_fxpts = fx.sanitize_points(
        np.concatenate(
            (-candidates, np.zeros((N,1)), candidates),
            axis=1),
        f, ef, Df,
        duplicates = rnn.duplicates_factory(W),
    )

    # Count union
    logger.log("(%d,%d): Sanitizing %d+%d=%d union points\n"%(
        N,sample, fxpts.shape[1], sign_change_fxpts.shape[1],
        fxpts.shape[1] + sign_change_fxpts.shape[1]))
    union_fxpts = fx.sanitize_points(
        np.concatenate(
            (fxpts, sign_change_fxpts),
            axis=1),
        f, ef, Df,
        duplicates = rnn.duplicates_factory(W),
    )

    logfile.close()

    print("(%d,%d) fxpt counts: new %d, old %d, union %d (status=%s)"%(
        N,sample,fxpts.shape[1], sign_change_fxpts.shape[1],
        union_fxpts.shape[1], status))
        
    results = {
        "status": status,
        "new count": fxpts.shape[1],
        "old count":  sign_change_fxpts.shape[1],
        "union": union_fxpts.shape[1],
        }

    with open(trialname(basename,N,sample)+".pkl",'w') as rf:
        pk.dump(results, rf)

def run_experiment(basename, network_sampling, timeout, num_procs=0):

    pool_args = []
    for (N,S) in network_sampling.items():
        for sample in range(S):
            # Sample network outside of pool for randomness
            _, _, _, W, V = rnn.make_known_fixed_points(N)
            pool_args.append((basename, N, sample, W, V, timeout))

    if num_procs > 0:

        num_procs = min(num_procs, mp.cpu_count())
        print("using %d processes..."%num_procs)
        pool = mp.Pool(processes=num_procs)
        pool.map(run_trial, pool_args)
        pool.close()
        pool.join()

    else:

        for pa in pool_args: run_trial(pa)

def plot_results(basename, network_sampling):
    network_sizes = network_sampling.keys()
    network_sizes.sort()
    colors = np.linspace(.75, 0., len(network_sizes)) # light to dark

    results = {}
    for (N,S) in network_sampling.items():
        for sample in range(S):
            try:
                with open(trialname(basename,N,sample)+".pkl",'r') as rf:
                    results[N,sample] = pk.load(rf)
            except:
                print("error: %d,%d"%(N,sample))

    handles = []
    mnx = 0
    pt.figure(figsize=(6,3))
    for i, N in enumerate(network_sizes):
        rgba = [colors[i]]*3 + [1]
        S = network_sampling[N]
        SSS = [s for s in range(S) if (N,s) in results]
        old_counts = [results[N,s]["old count"] for s in SSS]
        new_counts = [results[N,s]["new count"] for s in SSS]
        count_diff = [n-o for n,o in zip(new_counts,old_counts)]
        mnx = max(mnx, min(max(old_counts), max(new_counts)))
        # handles.append(pt.plot(old_counts, new_counts, 'o', fillstyle='none', markeredgecolor = rgba)[0])
        handles.append(pt.plot(old_counts, count_diff, 'o', fillstyle='none', markeredgecolor = rgba)[0])
        old_counts = [results[N,s]["old count"] for s in SSS
            if results[N,s]["status"]=="Terminated"]
        new_counts = [results[N,s]["new count"] for s in SSS
            if results[N,s]["status"]=="Terminated"]
        count_diff = [n-o for n,o in zip(new_counts,old_counts)]
        # pt.plot(old_counts, new_counts, '+', fillstyle='none', markeredgecolor = rgba)
        pt.plot(old_counts, count_diff, '+', fillstyle='none', markeredgecolor = rgba)
    # pt.plot([0, mnx], [0, mnx], linestyle='--',color=(0.85,)*3+(1,), zorder=-100)
    # pt.xscale('log')
    # pt.yscale('log')
    pt.legend(handles, ["N=%d"%N for N in network_sizes],loc="upper left")
    pt.xlabel("Old counts")
    # pt.ylabel("New counts")
    pt.ylabel("New counts - Old counts")
    pt.tight_layout()
    pt.show()
    
if __name__ == "__main__":

    basename = os.path.dirname(os.path.abspath(__file__)) + "/rnn_cand"

    # Maps network size: sample size
    network_sampling = {
        4: 20,
        8: 20,
        12: 20,
        16: 20,
        32: 10,
        64: 5,
        128: 5,
    }

    # run_experiment(basename, network_sampling, timeout=60*60, num_procs=5)
    plot_results(basename, network_sampling)
