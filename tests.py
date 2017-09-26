import unittest as ut
import numpy as np
import directional_fibers as df

class DirectionalFiberTestCase(ut.TestCase):
    def setUp(self):
        self.N = 2
        self.W = np.random.randn(self.N)
        self.fDf = lambda v: (
            (np.tanh(self.W.dot(v)) - v,
            (1-np.tanh(self.W.dot(v))**2)*self.W - np.eye(self.N)))
        self.x = 0.01*np.random.randn(self.N+1,1)
        self.c = np.random.randn(self.N,1)
        self.c = self.c/np.linalg.norm(self.c)
        self.max_solve_iterations = 2**5
        self.solve_tolerance = 10**-18
        self.mu = 1
        self.max_step_size = 1

    # @ut.skip("")
    def test_initial(self):
        x, residuals = df.refine_initial(
            self.fDf, self.x, self.c, self.max_solve_iterations, self.solve_tolerance)
        # print("Test initial:")
        print("")
        print("x, residuals")
        print(x.T)
        print(residuals)
        self.assertTrue(
            (residuals[-1] < self.solve_tolerance) or
            (len(residuals) <= self.max_solve_iterations))

    # @ut.skip("")
    def test_update_tangent(self):
        x, _ = df.refine_initial(
            self.fDf, self.x, self.c, self.max_solve_iterations, self.solve_tolerance)        
        _, Df = self.fDf(x[:self.N,:])
        DF = np.concatenate((Df, -self.c), axis=1)
        _,_,z = np.linalg.svd(DF)
        z = z[[self.N],:].T
        
        x = x + 0.001*z
        _, Df = self.fDf(x[:self.N,:])
        DF = np.concatenate((Df, -self.c), axis=1)
        z_new = df.update_tangent(DF, z)
        
        # print("Test update tangent:")
        print("")
        print("z, z_new")
        print(z.T)
        print(z_new.T)
        self.assertTrue(z.T.dot(z_new) > 0)

    # @ut.skip("")
    def test_compute_step_size(self):
        x, _ = df.refine_initial(
            self.fDf, self.x, self.c, self.max_solve_iterations, self.solve_tolerance)        
        _, Df = self.fDf(x[:self.N,:])
        DF = np.concatenate((Df, -self.c), axis=1)
        _,_,z = np.linalg.svd(DF)
        z = z[[self.N],:].T

        step_size = df.compute_step_size(self.mu, DF, z)
        print("")
        print("step_size")
        print(step_size) # sometimes = 1/(2mu) if all svs of DF > 1 (z gets the = 1)

    # @ut.skip("")
    def test_take_step(self):
        x, _ = df.refine_initial(
            self.fDf, self.x, self.c, self.max_solve_iterations, self.solve_tolerance)        
        _, Df = self.fDf(x[:self.N,:])
        DF = np.concatenate((Df, -self.c), axis=1)
        _,_,z = np.linalg.svd(DF)
        z = z[[self.N],:].T
        
        step_size = df.compute_step_size(self.mu, DF, z)
        if self.max_step_size is not None: step_size = min(step_size, self.max_step_size)

        x_new, residuals = df.take_step(self.fDf, self.c, z, x, step_size, self.max_solve_iterations, self.solve_tolerance)

        print("")
        print("x, x_new, residuals, num iters")
        print(x.T)
        print(x_new.T)
        print(residuals)
        print(len(residuals))
        self.assertTrue(z.T.dot(x_new-x) > 0)
        self.assertTrue(
            (len(residuals) <= self.max_solve_iterations) or
            (residuals[-1] < self.solve_tolerance))

def main():
    test_suite = ut.TestLoader().loadTestsFromTestCase(DirectionalFiberTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
    
if __name__ == "__main__":
    main()
