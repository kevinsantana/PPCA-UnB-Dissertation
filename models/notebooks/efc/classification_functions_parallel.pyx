import concurrent.futures
cimport cython
import itertools
import multiprocessing
import time

import numpy as np
cimport numpy as cnp

from dca_functions import *


DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t


def DefineCutoff(cnp.ndarray[cnp.int64_t, ndim=2] train_data, double[:] h_i, double[:,:] couplingmatrix, int Q):
    cdef int n_inst = train_data.shape[0]
    cdef int n_attr = train_data.shape[1]
    cdef double[:] energies = np.empty(n_inst, dtype=np.double)
    cdef int i, j, k, k_value, j_value
    cdef double e
    for i in range(n_inst):
        e = 0
        for j in range(n_attr-1):
            j_value = train_data[i,j]
            if j_value != (Q-1):
                for k in range(j,n_attr):
                    k_value = train_data[i,k]
                    if k_value != (Q-1):
                        e -= (couplingmatrix[j*(Q-1) + j_value, k*(Q-1) + k_value])
                e -= (h_i[j*(Q-1) + j_value])
        energies[i] = e
    energies = np.sort(energies, axis=None)
    cutoff = energies[int(energies.shape[0]*0.95)]
    return cutoff


def OneClassFit(cnp.ndarray[cnp.int64_t, ndim=2] data, int Q, double LAMBDA):
    cdef cnp.ndarray[double, ndim=2] sitefreq = Sitefreq(data, Q, LAMBDA)
    cdef cnp.ndarray[double, ndim=4] pairfreq = Pairfreq(data, sitefreq, Q, LAMBDA)
    cdef cnp.ndarray[double, ndim=2] couplingmatrix = Coupling(sitefreq, pairfreq, Q)
    cdef cnp.ndarray[double, ndim=1] h_i = LocalFields(couplingmatrix, sitefreq, Q)
    couplingmatrix = np.log(couplingmatrix)
    h_i = np.log(h_i)
    cdef double cutoff = DefineCutoff(data, h_i, couplingmatrix, Q)
    return couplingmatrix, h_i, cutoff, sitefreq, pairfreq


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def OneClassPredict(DTYPE_t[:,:] test_data, double[:,:] couplingmatrix, double[:] h_i, double cutoff, int Q):
    cdef int n_inst = test_data.shape[0]
    cdef int n_attr = test_data.shape[1]
    cdef double[:] energies = np.empty(n_inst, dtype=np.double)
    cdef DTYPE_t[:] predicted = np.empty(n_inst, dtype= 'int')
    cdef int i, j, k, k_value, j_value
    cdef double e
    for i in range(n_inst):
        e = 0
        for j in range(n_attr-1):
            j_value = test_data[i,j]
            if j_value != (Q-1):
                for k in range(j,n_attr):
                    k_value = test_data[i,k]
                    if k_value != (Q-1):
                        e -= (couplingmatrix[j*(Q-1) + j_value, k*(Q-1) + k_value])
                e -= (h_i[j*(Q-1) + j_value])
        predicted[i] = e > cutoff
        energies[i] = e
    return np.asarray(predicted), np.asarray(energies)


def FitClass(cnp.ndarray[DTYPE_t, ndim=2] subset, int Q, double LAMBDA):
    cdef cnp.ndarray[double, ndim=2] sitefreq = Sitefreq(subset, Q, LAMBDA)
    cdef cnp.ndarray[double, ndim=4] pairfreq = Pairfreq(subset, sitefreq, Q, LAMBDA)
    cdef cnp.ndarray[double, ndim=2] couplingmatrix = Coupling(sitefreq, pairfreq, Q)
    cdef cnp.ndarray[double, ndim=1] h_i = LocalFields(couplingmatrix, sitefreq, Q)
    couplingmatrix = np.log(couplingmatrix)
    h_i = np.log(h_i)
    cdef double cutoff = DefineCutoff(subset, h_i, couplingmatrix, Q)
    return h_i, couplingmatrix, cutoff


def MultiClassFit(cnp.ndarray[DTYPE_t, ndim=2] data, cnp.ndarray[DTYPE_t, ndim=1] labels, int Q, double LAMBDA):
    cdef int n_classes = np.unique(labels).shape[0]
    cdef cnp.ndarray[object, ndim=1] data_per_type = np.empty(n_classes, dtype=cnp.ndarray)
    cdef int label
    for indx, label in enumerate(np.unique(labels)):
      data_per_type[indx] = np.array([data[i,:] for i in range(data.shape[0]) if labels[i] == label])

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # futures_r = executor.map(FitClass, data_per_type, n_classes*[Q], n_classes*[LAMBDA])
        # result = concurrent.futures.wait(futures_r)
        # results.append(result)
        results = executor.map(FitClass, data_per_type, n_classes*[Q], n_classes*[LAMBDA])

    cdef cnp.ndarray[object, ndim=1] h_i_matrices = np.empty(n_classes, dtype=cnp.ndarray)
    cdef cnp.ndarray[object, ndim=1] coupling_matrices = np.empty(n_classes, dtype=cnp.ndarray)
    cdef cnp.ndarray[double, ndim=1] cutoffs_list = np.empty(n_classes, dtype='double')
    for indx, result in enumerate(results):
        h_i_matrices[indx] = result[0]
        coupling_matrices[indx] = result[1]
        cutoffs_list[indx] = result[2]

    return h_i_matrices, coupling_matrices, cutoffs_list


def PredictSubset(cnp.ndarray[DTYPE_t, ndim=2] test_data, cnp.ndarray[object, ndim=1] h_i_matrices, cnp.ndarray[object, ndim=1] coupling_matrices, cnp.ndarray[double, ndim=1] cutoffs_list, int Q, cnp.ndarray[DTYPE_t, ndim=1] train_labels):
    cdef int n_inst = test_data.shape[0]
    cdef int n_attr = test_data.shape[1]
    cdef int n_classes = h_i_matrices.shape[0]
    cdef cnp.ndarray predicted = np.empty(n_inst, dtype=int)
    cdef cnp.ndarray predicted_proba = np.empty((n_inst, n_classes), dtype=float)
    cdef int i, label, j, k, j_value, k_value, idx
    cdef double e, min_energy
    cdef cnp.ndarray[double, ndim=2] couplingmatrix
    cdef cnp.ndarray[double, ndim=1] h_i
    for i in range(n_inst):
        energies = []
        for label in range(n_classes):
          e = 0
          couplingmatrix = coupling_matrices[label]
          h_i = h_i_matrices[label]
          for j in range(n_attr-1):
              j_value = test_data[i,j]
              if j_value != (Q-1):
                  for k in range(j,n_attr):
                      k_value = test_data[i,k]
                      if k_value != (Q-1):
                          e -= (couplingmatrix[j*(Q-1) + j_value, k*(Q-1) + k_value])
                  e -= (h_i[j*(Q-1) + j_value])
          energies.append(e)

        predicted_proba[i, :] = energies
        min_energy = min(energies)
        idx = energies.index(min_energy)
        if min_energy < cutoffs_list[idx]:
            predicted[i] = np.unique(train_labels)[idx]
        else:
            predicted[i] = 100
    return predicted, predicted_proba


def MultiClassPredict(cnp.ndarray[DTYPE_t, ndim=2] test_data,
                      cnp.ndarray[object, ndim=1] h_i_matrices,
                      cnp.ndarray[object, ndim=1] coupling_matrices,
                      cnp.ndarray[double, ndim=1] cutoffs_list,
                      int Q,
                      cnp.ndarray[DTYPE_t, ndim=1] train_labels):
    cdef int n_jobs = 3
    cdef int chunk_size = test_data.shape[0] // n_jobs
    cdef int i
    cdef cnp.ndarray[object, ndim=1] data_frac = np.empty(n_jobs, dtype=cnp.ndarray)
    for i in range(n_jobs-1):
      data_frac[i] = test_data[i*chunk_size:(i+1)*chunk_size]
    data_frac[i+1] = test_data[(n_jobs-1)*chunk_size::]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        results = executor.map(PredictSubset, data_frac, n_jobs*[h_i_matrices], n_jobs*[coupling_matrices], n_jobs*[cutoffs_list], n_jobs*[Q], n_jobs*[train_labels])

    predicted = []
    predicted_proba = []
    for result in results:
        predicted += list(result[0])
        predicted_proba += list(result[1])

    return np.asarray(predicted), np.asarray(predicted_proba)
