"""
Measure global optimization performance of Levy function
"""

import sys, time
import numpy as np
import matplotlib.pyplot as pt
import multiprocessing as mp
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.logging_utilities as lu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
import dfibers.examples.levy as lv
from mpl_toolkits.mplot3d import Axes3D

def run_trial(args):
    basename, sample, timeout = args
    stop_time = time.clock() + timeout

    logfile = open("%s_s%d.log"%(basename,sample),"w")

    # Set up fiber arguments
    np.random.seed()
    v = 20*np.random.rand(2,1) - 10 # random point in domain
    c = lv.f(v) # direction at that point
    c = c + 0.1*np.random.randn(2,1) # perturb for more variability
    fiber_kwargs = {
        "f": lv.f,
        "ef": lv.ef,
        "Df": lv.Df,
        "compute_step_amount": lambda trace: (0.0001, 0),
        "v": v,
        "c": c,
        "stop_time": stop_time,
        "terminate": lambda trace: (np.fabs(trace.x[:-1]) > 10).any(),
        "max_solve_iterations": 2**5,
    }

    solve_start = time.clock()

    # Run in one direction
    solution = sv.fiber_solver(
        logger=lu.Logger(logfile).plus_prefix("+: "),
        **fiber_kwargs)
    X1 = np.concatenate(solution["Fiber trace"].points, axis=1)
    V1 = solution["Fixed points"]
    z = solution["Fiber trace"].z_initial
    # print("Status: %s\n"%solution["Fiber trace"].status)    
    
    # Run in other direction (negate initial tangent)
    solution = sv.fiber_solver(
        z= -z,
        logger=lu.Logger(logfile).plus_prefix("-: "),
        **fiber_kwargs)
    X2 = np.concatenate(solution["Fiber trace"].points, axis=1)
    V2 = solution["Fixed points"]
    # print("Status: %s\n"%solution["Fiber trace"].status)    

    # Join fiber segments
    fiber = np.concatenate((np.fliplr(X1), X2), axis=1)

    # Union solutions
    fxpts = fx.sanitize_points(
        np.concatenate((V1, V2), axis=1),
        f = lv.f,
        ef = lv.ef,
        Df = lv.Df,
        duplicates = lambda V, v: (np.fabs(V - v) < 10**-6).all(axis=0),
    )

    # Save results
    with open("%s_s%d.npz"%(basename,sample), 'w') as rf: np.savez(rf, **{
        "fxpts": fxpts,
        "fiber": fiber,
        "runtime": time.clock() - solve_start })

    logfile.close()

def run_experiment(basename, num_samples, timeout, num_procs=0):

    pool_args = []
    for sample in range(num_samples):
        pool_args.append((basename, sample, timeout))

    if num_procs > 0:

        num_procs = min(num_procs, mp.cpu_count())
        print("using %d processes..."%num_procs)
        pool = mp.Pool(processes=num_procs)
        pool.map(run_trial, pool_args)
        pool.close()
        pool.join()

    else:

        for pa in pool_args: run_trial(pa)

def compile_results(basename, num_samples):
    
    L = []
    F = []
    runtimes = []
    for sample in range(num_samples):
        with open("%s_s%d.npz"%(basename,sample), 'r') as rf: data = dict(np.load(rf))
        fxpts = data["fxpts"]
        Fs = np.fabs(lv.f(fxpts)).max(axis=0)
        Ls = lv.levy(fxpts)
        within = (np.fabs(fxpts) < 10).all(axis=0)
        mean_within = Ls[within].mean() if within.any() else np.nan
        print("sample %d: %d secs, %d solns, mean %f, mean within %f, min %f"%(
            sample, data["runtime"], len(Ls), Ls.mean(), mean_within, Ls.min()))
        L.append(Ls)
        F.append(Fs)
        runtimes.append(data["runtime"])
    
    counts = np.array([len(Ls) for Ls in L])
    bests = np.array([Ls.min() for Ls in L])
    resids = np.array([Fs.max() for Fs in F])
    runtimes = np.array(runtimes)
    print("avg count = %d, avg best = %f, avg resid = %f, best best = %f"%(
        counts.mean(), bests.mean(), resids.mean(), bests.min()))
    return counts, bests, runtimes

def plot_results(basename, num_samples, counts, bests, runtimes, timeout):

    ### Optimization order stats
    pt.figure(figsize=(5,4))
    pt.subplot(2,1,1)
    pt.plot(np.sort(bests), '-k.')
    pt.xlabel("Ordered samples")
    pt.ylabel("Best objective value")

    ##### Work complexity
    pt.subplot(2,1,2)
    terms = (runtimes < timeout)
    pt.plot(runtimes[terms], bests[terms], 'k+', markerfacecolor='none')
    pt.plot(runtimes[~terms], bests[~terms], 'ko', markerfacecolor='none')
    pt.legend(["terminated","timed out"])
    pt.xlabel("Runtime (seconds)")
    pt.ylabel("Best objective value")

    pt.tight_layout()
    pt.show()

    ### Fiber visuals

    pt.figure(figsize=(4,7))
    # objective fun
    X_surface, Y_surface = np.mgrid[-10:10:100j,-10:10:100j]
    L = lv.levy(np.array([X_surface.flatten(), Y_surface.flatten()])).reshape(X_surface.shape)
    ax_surface = pt.gcf().add_subplot(2,1,1,projection="3d")
    ax_surface.plot_surface(X_surface, Y_surface, L, linewidth=0, antialiased=False, color='gray')
    ax_surface.set_xlabel("v0")
    ax_surface.set_ylabel("v1")
    ax_surface.set_zlabel("levy(v)")
    ax_surface.view_init(azim=-80, elev=20)

    # fibers
    ax = pt.gcf().add_subplot(2,1,2)
    X_grid, Y_grid = np.mgrid[-10:10:60j,-10:10:60j]
    XY = np.array([X_grid.flatten(), Y_grid.flatten()])
    C_XY = lv.f(XY)
    ax.quiver(XY[0,:],XY[1,:],C_XY[0,:],C_XY[1,:],color=0.5*np.ones((1,3)),
        scale=10,units='xy',angles='xy')

    num_plot_samples = 3
    sort_idx = np.argsort(bests)
    plot_idx = [0] + list(np.random.permutation(num_samples)[:num_plot_samples-1])
    samples = sort_idx[plot_idx]
    # samples = [41,73,20] # all through global
    # samples = [41, 97, 11] # two through global
    # samples = [41, 49, 13] # two through global, one horiz not through
    # samples = [41, 46, 70] # one through global, one horiz
    # samples = [41, 96, 27] # two through global, one almost horiz
    samples = [41, 63, 28] # two through global, all interesting
    print("samples:")
    print(samples)
    for i,sample in enumerate(samples[::-1]):
        with open("%s_s%d.npz"%(basename,sample), 'r') as rf: data = dict(np.load(rf))
        fxpts = data["fxpts"]
        fiber = data["fiber"][:,::]
    
        L = lv.levy(fxpts).min()
        col = 0.5*float(num_plot_samples-i-1)/num_plot_samples
        print(sample,col)
        ax.plot(fiber[0],fiber[1],color=(col,col,col,1), linestyle='-', linewidth=1)
        pt.plot(fxpts[0],fxpts[1], 'o', color=(col,col,col,1))

    pt.xlabel("v0")
    pt.ylabel("v1",rotation=0)
    pt.yticks(np.linspace(-10,10,5))
    pt.xlim([-10,10])
    pt.ylim([-10,10])
    pt.tight_layout()
    pt.show()

if __name__ == "__main__":

    basename = "levy_opt"
    num_samples = 100
    num_plot_samples = 3
    timeout = 60*30
    num_procs = 10

    # run_experiment(basename, num_samples=num_samples, timeout=timeout, num_procs=num_procs)
    counts, bests, runtimes = compile_results(basename, num_samples)
    plot_results(basename, num_samples, counts, bests, runtimes, timeout)
