import numpy as np
import pandas as pd

from classification_functions import OneClassFit, OneClassPredict, MultiClassFit, MultiClassPredict

# prices array
efc_np = np.array([[323071], [323000], [323071], [323000], [323361], [323000], [323093], [323890], [323816], [323912]], dtype=np.int64)

# EFC's hyperparameters
Q = np.max(efc_np)
LAMBDA = 0.5 # pseudocount parameter

coupling, h_i, cutoff, _, _  = OneClassFit(efc_np, Q, LAMBDA)