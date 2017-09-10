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
    solve_tolerance = 10**-32
    
    x, F, i = df.refine_initial(fDf, x, c, max_solve_iterations, solve_tolerance)
    print("Test initial:")
    print(i)
    print(x.T)
    print(F.T)

def main():
    test_initial()

if __name__ == "__main__":
    main()
