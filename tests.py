import numpy as np
import directional_fibers as df

def test_initial():
    N = 2
    W = np.random.randn(N)
    fDf = lambda v: (np.tanh(W.dot(v)) - v, (1-np.tanh(W.dot(v))**2)*W - np.eye(N))
    
    x = 0.01*np.random.randn(N+1,1)
    c = np.random.randn(N,1)
    c = c/np.linalg.norm(c)    
    max_solve_iterations = 2**5
    solve_tolerance = 10**-18
    
    x, residuals = df.refine_initial(fDf, x, c, max_solve_iterations, solve_tolerance)
    print("Test initial:")
    print(x.T)
    print(residuals)

def test_update_tangent():
    N = 2
    W = np.random.randn(N)
    fDf = lambda v: (np.tanh(W.dot(v)) - v, (1-np.tanh(W.dot(v))**2)*W - np.eye(N))
    
    x = 0.01*np.random.randn(N+1,1)
    c = np.random.randn(N,1)
    c = c/np.linalg.norm(c)    

    max_solve_iterations = 2**5
    solve_tolerance = 10**-18    
    x, _ = df.refine_initial(fDf, x, c, max_solve_iterations, solve_tolerance)
    
    f, Df = fDf(x[:N,:])
    DF = np.concatenate((Df, -c), axis=1)
    _,_,z = np.linalg.svd(DF)
    z = z[[N],:].T
            
    z_new = df.update_tangent(DF, z)
    
    print("Test update tangent:")
    print(z.T)
    print(z_new.T)

def main():
    # test_initial()
    test_update_tangent()

if __name__ == "__main__":
    main()
