import numpy as np
from scipy import spatial


def weights(data: np.array, THETA: float):
    hamm_dist = spatial.distance.pdist(data, "hamming")
    weight_matrix = spatial.distance.squareform(hamm_dist < (1.0 - THETA))
    weight = 1.0 / (np.sum(weight_matrix, axis=1) + 1.0)
    return weight


def site_freq(data: np.array, Q: int, LAMBDA: float):
    n_attr = data.shape[1]
    sitefreq = np.empty((n_attr, Q), dtype="float")
    for i in range(n_attr):
        for aa in range(Q):
            sitefreq[i, aa] = np.sum(np.equal(data[:, i], aa))

    sitefreq /= data.shape[0]
    sitefreq = (1 - LAMBDA) * sitefreq + LAMBDA / Q
    return sitefreq


def cantor(x: np.array, y: np.array):
    return (x + y) * (x + y + 1) / 2 + y


def pair_freq(data: np.array, sitefreq: np.array, Q: int, LAMBDA: float):
    n_attr = data.shape[1]
    data_view = data
    pair_freq = np.zeros((n_attr, Q, n_attr, Q), dtype="float")
    pair_freq_view = pair_freq

    for i in range(n_attr):
        for j in range(n_attr):
            c = cantor(data[:, i], data[:, j])
            unique, aaIdx = np.unique(c, True)
            for x, item in enumerate(unique):
                pair_freq_view[i, data_view[aaIdx[x], i], j, data_view[aaIdx[x], j]] = (
                    np.sum(np.equal(c, item))
                )

    pair_freq /= data.shape[0]
    pair_freq = (1 - LAMBDA) * pair_freq + LAMBDA / (Q * Q)

    for i in range(n_attr):
        for am_i in range(Q):
            for am_j in range(Q):
                if am_i == am_j:
                    pair_freq[i, am_i, i, am_j] = sitefreq[i, am_i]
                else:
                    pair_freq[i, am_i, i, am_j] = 0.0
    return pair_freq


def local_fields(coupling_matrix: np.array, site_freq: np.array, Q: int):
    n_inst = site_freq.shape[0]
    fields = np.empty((n_inst * (Q - 1)), dtype="float")

    for i in range(n_inst):
        for ai in range(Q - 1):
            fields[i * (Q - 1) + ai] = site_freq[i, ai] / site_freq[i, Q - 1]
            for j in range(n_inst):
                for aj in range(Q - 1):
                    fields[i * (Q - 1) + ai] /= (
                        coupling_matrix[i * (Q - 1) + ai, j * (Q - 1) + aj]
                        ** site_freq[j, aj]
                    )
    return fields


def coupling(site_freq: np.array, pair_freq: np.array, Q: int):
    n_attr = site_freq.shape[0]
    corr_matrix = np.empty(((n_attr) * (Q - 1), (n_attr) * (Q - 1)), dtype="float")

    for i in range(n_attr):
        for j in range(n_attr):
            for am_i in range(Q - 1):
                for am_j in range(Q - 1):
                    corr_matrix[i * (Q - 1) + am_i, j * (Q - 1) + am_j] = (
                        pair_freq[i, am_i, j, am_j]
                        - site_freq[i, am_i] * site_freq[j, am_j]
                    )

    inv_corr = np.linalg.inv(corr_matrix)
    coupling_matrix = np.exp(np.negative(inv_corr))
    return coupling_matrix
