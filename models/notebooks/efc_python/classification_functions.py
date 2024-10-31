import concurrent
import numpy as np
from tqdm import tqdm

from efc_python.dca_functions import coupling, local_fields, pair_freq, site_freq


def define_cutoff(train_data: np.array, h_i, coupling_matrix, Q: int):
    n_inst = train_data.shape[0]
    n_attr = train_data.shape[1]
    energies = np.empty(n_inst, dtype=np.double)

    for i in tqdm(range(n_inst)):
        e = 0
        for j in range(n_attr - 1):
            j_value = train_data[i, j]
            if j_value != (Q - 1):
                for k in range(j, n_attr):
                    k_value = train_data[i, k]
                    if k_value != (Q - 1):
                        e -= coupling_matrix[
                            j * (Q - 1) + j_value, k * (Q - 1) + k_value
                        ]
                e -= h_i[j * (Q - 1) + j_value]
        energies[i] = e

    energies = np.sort(energies, axis=None)
    cutoff = energies[int(energies.shape[0] * 0.95)]

    return cutoff


def one_class_fit(data: np.array, Q: int, LAMBDA: float):
    s_freq = site_freq(data, Q, LAMBDA)
    p_freq = pair_freq(data, s_freq, Q, LAMBDA)
    c_matrix = coupling(s_freq, p_freq, Q)
    h_i = local_fields(c_matrix, s_freq, Q)
    c_matrix = np.log(c_matrix)
    h_i = np.log(h_i)
    cutoff = define_cutoff(data, h_i, c_matrix, Q)
    return c_matrix, h_i, cutoff, s_freq, p_freq


def one_class_predict(test_data, coupling_matrix, h_i, cutoff, Q):
    n_inst = test_data.shape[0]
    n_attr = test_data.shape[1]
    energies = np.empty(n_inst, dtype=np.double)
    predicted = np.empty(n_inst, dtype="int")

    for i in tqdm(range(n_inst)):
        e = 0
        for j in range(n_attr - 1):
            j_value = test_data[i, j]
            if j_value != (Q - 1):
                for k in range(j, n_attr):
                    k_value = test_data[i, k]
                    if k_value != (Q - 1):
                        e -= coupling_matrix[
                            j * (Q - 1) + j_value, k * (Q - 1) + k_value
                        ]
                e -= h_i[j * (Q - 1) + j_value]
        predicted[i] = e > cutoff
        energies[i] = e
    return np.asarray(predicted), np.asarray(energies)


def fit_class(subset: np.array, Q: int, LAMBDA: float):
    s_freq = site_freq(subset, Q, LAMBDA)
    p_freq = pair_freq(subset, s_freq, Q, LAMBDA)
    c_matrix = coupling(s_freq, p_freq, Q)
    h_i = local_fields(c_matrix, s_freq, Q)
    c_matrix = np.log(c_matrix)
    h_i = np.log(h_i)
    cutoff = define_cutoff(subset, h_i, c_matrix, Q)
    return h_i, c_matrix, cutoff


def multi_class_fit(data: np.array, labels: np.array, Q: int, LAMBDA: float):
    n_classes = np.unique(labels).shape[0]
    data_per_type = np.empty(n_classes, dtype=np.ndarray)

    for indx, label in enumerate(np.unique(labels)):
        data_per_type[indx] = np.array(
            [data[i, :] for i in range(data.shape[0]) if labels[i] == label]
        )

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # futures_r = executor.map(FitClass, data_per_type, n_classes*[Q], n_classes*[LAMBDA])
        # result = concurrent.futures.wait(futures_r)
        # results.append(result)
        results = executor.map(
            fit_class, data_per_type, n_classes * [Q], n_classes * [LAMBDA]
        )

    h_i_matrices = np.empty(n_classes, dtype=np.ndarray)
    c_matrices = np.empty(n_classes, dtype=np.ndarray)
    cutoffs_list = np.empty(n_classes, dtype="double")

    for indx, result in enumerate(results):
        h_i_matrices[indx] = result[0]
        c_matrices[indx] = result[1]
        cutoffs_list[indx] = result[2]

    return h_i_matrices, c_matrices, cutoffs_list


def predict_subset(
    test_data: np.array,
    h_i_matrices: np.array,
    coupling_matrices: np.array,
    cutoffs_list: np.array,
    Q: int,
    train_labels: np.array,
):
    n_inst = test_data.shape[0]
    n_attr = test_data.shape[1]
    n_classes = h_i_matrices.shape[0]
    predicted = np.empty(n_inst, dtype=int)
    predicted_proba = np.empty((n_inst, n_classes), dtype=float)

    for i in tqdm(range(n_inst)):
        energies = []
        for label in range(n_classes):
            e = 0
            couplingmatrix = coupling_matrices[label]
            h_i = h_i_matrices[label]
            for j in range(n_attr - 1):
                j_value = test_data[i, j]
                if j_value != (Q - 1):
                    for k in range(j, n_attr):
                        k_value = test_data[i, k]
                        if k_value != (Q - 1):
                            e -= couplingmatrix[
                                j * (Q - 1) + j_value, k * (Q - 1) + k_value
                            ]
                    e -= h_i[j * (Q - 1) + j_value]
            energies.append(e)

        predicted_proba[i, :] = energies
        min_energy = min(energies)
        idx = energies.index(min_energy)
        if min_energy < cutoffs_list[idx]:
            predicted[i] = np.unique(train_labels)[idx]
        else:
            predicted[i] = 100
    return predicted, predicted_proba


def multi_class_predict(
    test_data: np.array,
    h_i_matrices: np.array,
    coupling_matrices: np.array,
    cutoffs_list: np.array,
    Q: int,
    train_labels: np.array,
):
    n_jobs = 3
    chunk_size = test_data.shape[0] // n_jobs
    data_frac = np.empty(n_jobs, dtype=np.ndarray)

    for i in range(n_jobs - 1):
        data_frac[i] = test_data[i * chunk_size : (i + 1) * chunk_size]
    data_frac[i + 1] = test_data[(n_jobs - 1) * chunk_size : :]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        results = executor.map(
            predict_subset,
            data_frac,
            n_jobs * [h_i_matrices],
            n_jobs * [coupling_matrices],
            n_jobs * [cutoffs_list],
            n_jobs * [Q],
            n_jobs * [train_labels],
        )

    predicted = []
    predicted_proba = []

    for result in results:
        predicted += list(result[0])
        predicted_proba += list(result[1])

    return np.asarray(predicted), np.asarray(predicted_proba)
