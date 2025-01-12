import numpy as np

from classification_functions import (
    multi_class_fit,
    multi_class_predict,
    one_class_fit,
    one_class_predict,
)
from generic_discretize import discretize, get_intervals


class EFC:
    def __init__(self, Q, LAMBDA):
        self.Q = Q
        self.LAMBDA = LAMBDA

    def _prepare_for_fit(
        self,
        X_train: np.array,
        X_test: np.array,
        y_train: np.array,
        drop_abnormal=False,
    ):
        """
        Prepare a dataset to be EFC compatible.
        1. get discretization intervals from train set
        2. discretize train and test
        OPTIONAL
        3. find abnormal samples indexes in the training set
        4. remove abnormal samples from training (EFC trains with only benign instances)
        5. remove the corresponding abonrmal training targets
        """

        intervals = get_intervals(X_train, 10)
        X_train = discretize(X_train, intervals)
        X_test = discretize(X_test, intervals)

        if drop_abnormal and y_train:
            idx_abnormal = np.where(y_train == 1)[0]
            X_train.drop(idx_abnormal, axis=0, inplace=True)
            y_train.drop(idx_abnormal, axis=0, inplace=True)

        return X_train, X_test, y_train

    def fit_one_class(
        self,
        X_train: np.array,
        X_test: np.array,
        y_train: np.array,
        drop_abnormal=False,
        Q=None,
        LAMBDA=None,
    ):
        X_train, X_test, y_train = self._prepare_for_fit(
            X_train=X_train, X_test=X_test, y_train=y_train, drop_abnormal=drop_abnormal
        )

        c_matrix, h_i, cutoff, s_freq, p_freq = one_class_fit(
            data=X_train, Q=self.Q or Q, LAMBDA=self.LAMBDA or LAMBDA
        )
        self.coupling = c_matrix
        self.h_i = h_i
        self.cutoff = cutoff
        self.s_freq = s_freq
        self.p_freq = p_freq

    def decision_function_one_class(self, X_test: np.array, coupling, h_i, cutoff, Q):
        y_predicted, energies = one_class_predict(
            test_data=X_test,
            coupling_matrix=self.coupling or coupling,
            h_i=self.h_i or h_i,
            cutoff=self.cutoff or cutoff,
            Q=self.Q or Q,
        )
        self.y_predicted = y_predicted
        self.energies = energies
        return y_predicted

    @property
    def decision_scores_(self):
        return NotImplementedError

    def fit_multi_class(
        self,
        X_train: np.array,
        X_test: np.array,
        y_train: np.array,
        drop_abnormal=False,
        Q=None,
        LAMBDA=None,
    ):
        X_train, X_test, y_train = self._prepare_for_fit(
            X_train=X_train, X_test=X_test, y_train=y_train, drop_abnormal=drop_abnormal
        )

        coupling_matrices, h_i_matrices, cutoffs_list = multi_class_fit(
            data=X_train, labels=y_train, Q=self.Q, LAMBDA=self.LAMBDA
        )
        self.coupling_matrices = coupling_matrices
        self.h_i_matrices = h_i_matrices
        self.cutoffs_list = cutoffs_list

        y_predicted, y_predicted_proba = multi_class_predict(
            test_data=X_test,
            h_i_matrices=self.h_i_matrices,
            coupling_matrices=self.coupling_matrices,
            cutoffs_list=self.cutoffs_list,
            Q=self.Q,
            train_labels=y_train,
        )
        self.y_predicted = y_predicted
        self.y_predicted_proba = y_predicted_proba
