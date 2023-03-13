import time
import unittest as ut
import numpy as np
import dfibers.numerical_utilities as nu
import dfibers.traversal as tv
import dfibers.fixed_points as fx
import dfibers.solvers as sv
import dfibers.examples.rnn as rnn
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

VERBOSE = False

class FixedPointsTestCase(ut.TestCase):
    def setUp(self):
        self.N = 10
        self.P = 100
        self.K = 3
        self.noise = 5
        self.E = lambda V, u: (np.fabs(V - u) < self.noise*nu.eps(V)).all(axis=0)
    def get_test_points(self):
        """
        Construct a set of P*K test points with at most K "unique" members.
        returns a numpy.array V, where V[:,p] is the p^{th} point.
        """
        # make P copies of K distinct, random points
        V = np.tile(np.random.rand(self.N,self.K),(1,self.P))
        # shuffle randomly
        V = V[:,np.random.permutation(self.K*self.P)]
        # perterb by a small multiple of machine precision
        V = V + np.floor(self.noise*np.random.rand(self.N,self.K*self.P))*nu.eps(V)
        return V
    def test_get_connected_components(self):
        """
        Sanity check for get_connected_components
        """
        V = self.get_test_points()
        components = fx.get_connected_components(V, self.E)
        self.assertTrue(len(np.unique(components)) <= self.K)
    def test_get_unique_points(self):
        """
        Sanity check for get_unique_points
        """
        V = self.get_test_points()
        U = fx.get_unique_points(V, self.E)
        self.assertTrue(U.shape[1] <= self.K)
        for p in range(V.shape[1]):
            noise = np.fabs(U - V[:,[p]]).max(axis=0)
            self.assertTrue(noise.min() < (self.noise*nu.eps(V[:,p])).max())

class RNNFixedPointsTestCase(ut.TestCase):
    def setUp(self):
        self.N = 10
        self.P = 5
        self.f, self.Df, self.ef, self.W, self.V = rnn.make_known_fixed_points(self.N)
        self.noise = 5
        # self.duplicates = lambda V, u: (np.fabs(V - u) < 2*self.noise*nu.eps(V)).all(axis=0)
        self.duplicates = rnn.duplicates_factory(self.W)
    def get_test_points(self):
        """
        Construct a set of P*K test points based on K known fixed points
        returns a numpy.array V, where V[:,p] is the p^{th} point.
        """
        # make P copies of K known points
        V = np.tile(self.V, (1,self.P))
        # shuffle randomly
        V = V[:,np.random.permutation(V.shape[1])]
        # perterb by a small multiple of machine precision
        V = V + np.floor(self.noise*np.random.rand(*V.shape))*nu.eps(V)
        return V
    def test_sanitize_points(self):
        """
        Sanity check for refine_points
        """
        V = self.get_test_points()
        U = fx.sanitize_points(V, self.f, self.ef, self.Df, self.duplicates)
        if VERBOSE: print('')
        if VERBOSE: print(V.shape)
        if VERBOSE: print(U.shape)
        if VERBOSE: print(self.V.shape)
        self.assertTrue(U.shape[1] == self.V.shape[1])
        for p in range(self.V.shape[1]):
            # noise = np.fabs(U - self.V[:,[p]]).max(axis=0)
            # self.assertTrue(noise.min() < (self.noise*nu.eps(self.V[:,p])).max())
            self.assertTrue(self.duplicates(U, self.V[:,[p]]).any())

class RNNLocalSolverTestCase(ut.TestCase):
    def setUp(self):
        self.N = 3
        self.P = 5
        self.f, self.Df, self.ef, self.W, self.V = rnn.make_known_fixed_points(self.N)
        self.sampler = rnn.sampler_factory(self.W)
        self.qg = rnn.qg_factory(self.W)
        self.H = rnn.H_factory(self.W)
        self.duplicates = rnn.duplicates_factory(self.W)
    def test_local_solver(self):
        result = sv.local_solver(
            self.sampler,
            self.f,
            self.qg,
            self.H,
            max_repeats=500,
        )
        V = result["Optima"]
        U = fx.sanitize_points(V, self.f, self.ef, self.Df, self.duplicates)
        if VERBOSE: print("This test should succeed with high probability")
        self.assertTrue(U.shape[1] >= self.V.shape[1])
        for p in range(self.V.shape[1]):
            self.assertTrue(self.duplicates(U, self.V[:,[p]]).any())

class FiberTraceTestCase(ut.TestCase):

    # @ut.skip("")
    def test_halve_points(self):
        trace = tv.FiberTrace(None)
        alpha = [3, 2, 1, .5, .4, 1, 2, 3, 2, 1, .5, -1, -2, -3]
        #        _, _, _,  c,  c, c, _, _, _, c,  c,  c,  _,  _ candidates
        #        k, _, k,  k,  k, k, _, k, _, k,  k,  k,  k,  _ every other non
        #        k, k, k,  k,  k, k, _, k, _, k,  k,  k,  k,  k leading and last
        for a in alpha:
            trace.points.append(np.array([[0,0,a]]).T)
            trace.tangents.append(None)
            trace.residuals.append(None)
            trace.step_amounts.append(None)
            trace.step_data.append(None)

        trace.halve_points()
        if VERBOSE: print([p[-1,0] for p in trace.points])
        if VERBOSE: print(len(trace.points))

        self.assertTrue(len(trace.points) == 12)

class RNNDirectionalFiberTestCase(ut.TestCase):
    def setUp(self):
        self.N = 2
        self.W = 1.25*np.eye(self.N,self.N) + 0.1*np.random.randn(self.N, self.N)
        self.f = rnn.f_factory(self.W)
        self.Df = rnn.Df_factory(self.W)
        self.ef = rnn.ef_factory(self.W)
        self.compute_step_amount = rnn.compute_step_amount_factory(self.W)
        self.x = 0.01*np.random.randn(self.N+1,1)
        self.c = np.random.randn(self.N,1)
        self.c = self.c/np.linalg.norm(self.c)
        self.max_solve_iterations = 2**5
        self.max_step_size = 1

    # @ut.skip("")
    def test_initial(self):
        x, residuals = tv.refine_initial(
            self.f, self.Df, self.ef, self.x, self.c, self.max_solve_iterations)
        # if VERBOSE: print("Test initial:")
        if VERBOSE: print("")
        if VERBOSE: print("x, residuals")
        if VERBOSE: print(x.T)
        if VERBOSE: print(residuals)
        self.assertTrue(
            (self.f(x[:-1,:]) < self.ef(x[:-1,:])).all() or
            (len(residuals) <= self.max_solve_iterations))

    # @ut.skip("")
    def test_update_tangent(self):
        x, _ = tv.refine_initial(
            self.f, self.Df, self.ef, self.x, self.c, self.max_solve_iterations)
        DF = np.concatenate((self.Df(x[:self.N])[0], -self.c), axis=1)
        _,_,z = np.linalg.svd(DF)
        z = z[[self.N],:].T
        
        x = x + 0.001*z
        DF = np.concatenate((self.Df(x[:self.N])[0], -self.c), axis=1)
        z_new = tv.compute_tangent(DF, z)
        
        # if VERBOSE: print("Test update tangent:")
        if VERBOSE: print("")
        if VERBOSE: print("z, z_new")
        if VERBOSE: print(z.T)
        if VERBOSE: print(z_new.T)
        self.assertTrue(z.T.dot(z_new) > 0)

    # @ut.skip("")
    def test_compute_step_amount_size(self):
        x, _ = tv.refine_initial(
            self.f, self.Df, self.ef, self.x, self.c, self.max_solve_iterations)
        DF = np.concatenate((self.Df(x[:self.N])[0], -self.c), axis=1)
        _,_,z = np.linalg.svd(DF)
        z = z[[self.N],:].T

        trace = tv.FiberTrace(self.c)
        trace.DF = DF
        trace.z = z

        step_size, sv_min, critical = self.compute_step_amount(trace)
        if VERBOSE: print("")
        if VERBOSE: print("step_size, sv_min, critical")
        if VERBOSE: print(step_size, sv_min, critical) # sometimes = 1/(2mu) if all svs of DF > 1 (z gets the = 1)

    # @ut.skip("")
    def test_take_step(self):
        x, _ = tv.refine_initial(
            self.f, self.Df, self.ef, self.x, self.c, self.max_solve_iterations)
        DF = np.concatenate((self.Df(x[:self.N])[0], -self.c), axis=1)
        _,_,z = np.linalg.svd(DF)
        z = z[[self.N],:].T
        
        trace = tv.FiberTrace(self.c)
        trace.DF = DF
        trace.z = z

        step_size, step_data, critical = self.compute_step_amount(trace)
        if self.max_step_size is not None: step_size = min(step_size, self.max_step_size)

        x_new, residuals = tv.take_step(
            self.f, self.Df, self.ef, self.c, z, x,
            step_size, self.max_solve_iterations)

        if VERBOSE: print("")
        if VERBOSE: print("x, x_new, residuals, num iters")
        if VERBOSE: print(x.T)
        if VERBOSE: print(x_new.T)
        if VERBOSE: print(residuals)
        if VERBOSE: print(len(residuals))
        self.assertTrue(z.T.dot(x_new-x) > 0)
        self.assertTrue(
            (self.f(x[:-1,:]) < self.ef(x[:-1,:])).all() or
            (len(residuals) <= self.max_solve_iterations+1))
    
    # @ut.skip("")
    def test_early_term(self):
        if VERBOSE: print("")
        for max_traverse_steps in range(1,5):
            result = tv.traverse_fiber(
                self.f,
                self.Df,
                self.ef,
                self.compute_step_amount,
                v=self.x[:self.N,:],
                c=self.c,
                max_traverse_steps=max_traverse_steps,
                max_solve_iterations=self.max_solve_iterations,
                )
                
            if VERBOSE: print("max, len(X):")
            if VERBOSE: print(max_traverse_steps, len(result.points))
            self.assertTrue(len(result.points) <= max_traverse_steps+1)
        run_time = 2
        start_time = time.perf_counter()
        result = tv.traverse_fiber(
            self.f,
            self.Df,
            self.ef,
            self.compute_step_amount,
            v=self.x[:self.N,:],
            c=self.c,
            stop_time=start_time + run_time,
            max_solve_iterations=self.max_solve_iterations,
            )
        end_time = time.perf_counter()
        if VERBOSE: print("start, run, end")
        if VERBOSE: print(start_time, run_time, end_time)
        self.assertTrue(end_time > start_time + run_time and end_time < start_time + run_time + 1)

    # @ut.skip("")
    def test_terminate(self):
        result = tv.traverse_fiber(
            self.f,
            self.Df,
            self.ef,
            self.compute_step_amount,
            v=self.x[:self.N,:],
            c=self.c,
            terminate=lambda trace: True,
            max_traverse_steps=2,
            max_solve_iterations=self.max_solve_iterations,
            )
        self.assertTrue(result.status == "Terminated")
        result = tv.traverse_fiber(
            self.f,
            self.Df,
            self.ef,
            self.compute_step_amount,
            v=self.x[:self.N,:],
            c=self.c,
            terminate=rnn.terminate_factory(self.W, self.c),
            max_traverse_steps=10000,
            max_solve_iterations=self.max_solve_iterations,
            )
        self.assertTrue(result.status == "Terminated")

    # @ut.skip("")
    def test_ef(self):
        ef = rnn.ef_factory(self.W)
        if VERBOSE: print()
        if VERBOSE: print("ef(0):")
        if VERBOSE: print(ef(np.zeros((self.N,1))).max())
        self.assertTrue(ef(np.zeros((self.N,1))).max() < 1**-100)
        if VERBOSE: print("ef(1):")
        if VERBOSE: print(ef(np.ones((self.N,1))).max())
        self.assertTrue(ef(np.zeros((self.N,1))).max() < 1**-10)

    # @ut.skip("")
    def test_traverse_fiber(self):
        result = tv.traverse_fiber(
            self.f,
            self.Df,
            self.ef,
            self.compute_step_amount,
            v=self.x[:self.N,:],
            c=self.c,
            terminate=rnn.terminate_factory(self.W, self.c),
            max_traverse_steps=1000,
            max_solve_iterations=self.max_solve_iterations,
            )
        X = np.concatenate(result.points,axis=1)
        V = X[:-1,:]
        C = self.f(V)
        self.assertTrue((np.fabs(X[[-1],:]*self.c - C) < 0.001).all())

def main():
    test_suite = ut.TestLoader().loadTestsFromTestCase(RNNDirectionalFiberTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
    test_suite = ut.TestLoader().loadTestsFromTestCase(FixedPointsTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
    test_suite = ut.TestLoader().loadTestsFromTestCase(RNNFixedPointsTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
    test_suite = ut.TestLoader().loadTestsFromTestCase(RNNLocalSolverTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
    test_suite = ut.TestLoader().loadTestsFromTestCase(FiberTraceTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
    
if __name__ == "__main__": main()
