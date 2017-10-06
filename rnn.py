"""
Basic recurrent neural network model with activation rule:
    v[t+1] = np.tanh(W.dot(v[t]))
"""
import numpy as np
import numerical_utilities as nu

# Constructs the fDf function for a given W
def fDf_factory(W):
    I = np.eye(W.shape[0])
    def fDf(v):
        f = np.tanh(W.dot(v)) - v
        Df = (1-np.tanh(W.dot(v))**2)*W - I
        return f, Df
    return fDf

# Constructs the compute_step_size function for a given W
def compute_step_size_factory(W):
    def compute_step_size():
        pass

# Constructs the compute_step_size function for a given invertible W
def compute_step_size_factory(W):
    mu = np.sqrt(16./27.) * np.linalg.norm(W) * min(np.linalg.norm(W), np.sqrt((W*W).sum(axis=1)).max())
    def compute_step_size(x, DF, z):
        DG = np.concatenate((DF, z.T), axis=0)
        sv_min = nu.minimum_singular_value(DG)
        step_size = sv_min / (4. * mu)
        return step_size, sv_min
    return compute_step_size
