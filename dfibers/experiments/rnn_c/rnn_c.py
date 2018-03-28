"""
Probes existence of a single c that locates all fixed points of RNN
"""
import sys, time
import multiprocessing as mp
import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import dfibers.logging_utilities as lu
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.examples.rnn as rnn

def run_trials(args):
    basename, proc, N, timeout = args

    # Reseed randomness, start logging, start timing
    np.random.seed()
    logfile = open("%s_p%d_N%d.log"%(basename,proc,N), "w")
    logger = lu.Logger(logfile)
    stop_time = time.clock()+timeout

    # Keep sampling until timeout
    for sample in it.count(0):

        # Sample new network
        f, Df, ef, W, V = rnn.make_known_fixed_points(N)
        dups = rnn.duplicates_factory(W)
        results = {}
        data = {"W": W}

        # Keep fiber-solving until some c finds all V or timeout
        for trial in it.count(0):
            print("proc %d, N %d, sample %d, trial %d"%(proc,N,sample,trial))
    
            # Run fiber solver
            start_time = time.clock()
            fxpts, solution = rnn.run_fiber_solver(W,
                # max_traverse_steps = 2**15,
                stop_time = stop_time,
                logger=logger.plus_prefix("(%d,%d,%d): "%(N,sample,trial)))
            run_time = time.clock() - start_time
    
            # Update union of fxpts
            V = fx.sanitize_points(
                np.concatenate((fxpts, V), axis=1),
                f, ef, Df,
                duplicates = dups,
            )
    
            # Save results
            trace = solution["Fiber trace"]
            results[trial] = {
                "status": trace.status,
                "seconds": run_time,
                "|fxpts|": fxpts.shape[1],
                "|V|": V.shape[1],
                }
            results["trials"] = trial+1
            data[str((trial,"c"))] = trace.c
            data[str((trial,"fxpts"))] = fxpts
            data["V"] = V

            sample_basename = "%s_p%d_N%d_s%d"%(basename,proc,N,sample)
            with open(sample_basename+".pkl",'w') as rf: pk.dump(results, rf)
            with open(sample_basename+".npz",'w') as df: np.savez(df, **data)
    
            print("%d,%d,%d,%d: status %s, |fxpts|=%d, |V|=%d"%(
                proc,N,sample,trial,trace.status,fxpts.shape[1],V.shape[1]))
    
            # Done if current c found full union
            if fxpts.shape[1] == V.shape[1]: break
            
            # Done if timeout
            if time.clock() > stop_time: break
    
        print("%d,%d,%d: %d trials"%(proc,N,sample,trial+1))
        if time.clock() > stop_time: break

    proc_basename = "%s_p%d_N%d"%(basename,proc,N)
    with open(proc_basename+".pkl",'w') as rf:
        pk.dump({"num_samples":sample+1}, rf)

    logfile.close()

def run_experiment(basename, network_sizes, timeout, num_procs=0):

    pool_args = []
    for p,N in enumerate(network_sizes):
        pool_args.append((basename, p, N, timeout))

    if num_procs > 0:

        num_procs = min(num_procs, mp.cpu_count())
        print("using %d processes..."%num_procs)
        pool = mp.Pool(processes=num_procs)
        pool.map(run_trials, pool_args)
        pool.close()
        pool.join()

    else:

        for pa in pool_args: run_trials(pa)

def plot_results(basename, network_sizes):

    # results = {}
    # data = {"W": W}
    # results[trial] = {
    #     "status": trace.status,
    #     "seconds": run_time,
    #     "|fxpts|": fxpts.shape[1],
    #     "|V|": V.shape[1],
    #     }
    # results["trials"] = trial+1
    # data[str((trial,"c"))] = trace.c
    # data[str((trial,"fxpts"))] = fxpts
    # data["V"] = V
    # sample_basename = "%s_p%d_N%d_s%d"%(basename,proc,N,sample)
    # with open(sample_basename+".pkl",'w') as rf: pk.dump(results, rf)
    # with open(sample_basename+".npz",'w') as df: np.savez(df, **data)
    # proc_basename = "%s_p%d_N%d"%(basename,proc,N)
    # with open(proc_basename+".pkl",'w') as rf:
    #     pk.dump({"num_samples":sample+1}, rf)

    print("*****************")
    for p,N in enumerate(network_sizes):
        with open("%s_p%d_N%d"%(basename,p,N)+".pkl",'r') as rf:
            num_samples = pk.load(rf)["num_samples"]
        for sample in range(num_samples):
            sample_basename = "%s_p%d_N%d_s%d"%(basename,p,N,sample)
            with open(sample_basename+".pkl",'r') as rf: results = pk.load(rf)
            # with open(sample_basename+".npz",'r') as df: data = np.load(df)

            num_trials = results["trials"]
            for trial in range(num_trials):
                print("%d,%d,%d,%d: status %s, %f seconds, |fxpts|=%d, |V|=%d"%(
                    p,N,sample,trial,
                    results[trial]["status"],
                    results[trial]["seconds"],
                    results[trial]["|fxpts|"],
                    results[trial]["|V|"]))
            
    #         pt.plot(range(num_trials), [
    #             # float(results[t]["|fxpts|"])/float(results[num_trials-1]["|V|"])
    #             float(results[t]["|fxpts|"])
    #             for t in range(num_trials)],'-k')
    #         pt.plot(range(num_trials), [
    #             # float(results[t]["|V|"])/float(results[num_trials-1]["|V|"])
    #             float(results[t]["|V|"])
    #             for t in range(num_trials)],'--k')
    # pt.show()

if __name__ == "__main__":

    basename = "rnn_c"
    network_sizes = [3,4,5,6,7]+[3,4,5,6,7]
    # num_procs = 5
    timeout = 60*60*18
    run_experiment(basename, network_sizes, timeout, num_procs=len(network_sizes))

    plot_results(basename, network_sizes)
