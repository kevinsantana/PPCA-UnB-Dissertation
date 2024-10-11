import numpy as np
from scipy import spatial


def Weights(data: np.array, THETA: float):
    hamm_dist = spatial.distance.pdist(data, 'hamming')
    weight_matrix = spatial.distance.squareform(hamm_dist < (1.0 - THETA))
    weight = 1.0 / (np.sum(weight_matrix, axis = 1) + 1.0)
    return weight


def Sitefreq(data: np.array, Q: int, LAMBDA: float):
    n_attr = data.shape[1]
    sitefreq = np.empty((n_attr, Q), dtype='float')
    for i in range(n_attr):
        for aa in range(Q):
            sitefreq[i, aa] = np.sum(np.equal(data[:,i],aa))

    sitefreq /= data.shape[0]
    sitefreq = (1 - LAMBDA) * sitefreq + LAMBDA / Q
    return sitefreq
