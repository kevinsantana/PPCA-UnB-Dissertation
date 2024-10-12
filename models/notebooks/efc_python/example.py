import numpy as np


from classification_functions import (
    multi_class_fit,
    multi_class_predict,
    one_class_fit,
    one_class_predict,
)
from generic_discretize import discretize, get_intervals
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split


def single_class_efc():
    """Usage example of Single-class EFC"""
    data = load_breast_cancer(
        as_frame=True
    )  # load toy dataset from scikit-learn (binary targets)

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.33, shuffle=False
    )  # split dataset into training and testing sets

    intervals = get_intervals(
        X_train, 10
    )  # get discretization intervals from train set

    X_train = discretize(X_train, intervals)  # discretize train
    X_test = discretize(X_test, intervals)  # discretize test

    idx_abnormal = np.where(y_train == 1)[
        0
    ]  # find abnormal samples indexes in the training set

    X_train.drop(
        idx_abnormal, axis=0, inplace=True
    )  # remove abnormal samples from training (EFC trains with only benign instances)

    y_train.drop(
        idx_abnormal, axis=0, inplace=True
    )  # remove the corresponding abonrmal training targets

    # EFC's hyperparameters
    Q = X_test.values.max()
    LAMBDA = 0.5  # pseudocount parameter

    coupling, h_i, cutoff, _, _ = one_class_fit(
        np.array(X_train), Q, LAMBDA
    )  # train model

    y_predicted, energies = one_class_predict(
        np.array(X_test), coupling, h_i, cutoff, Q
    )  # test model

    # colect results
    print("Single-class results")
    print("confusion_matrix", confusion_matrix(y_test, y_predicted))
    return y_test, y_predicted


def multi_class_efc():
    """Usage example of Multi-class EFC"""
    data = load_wine(
        as_frame=True
    )  # load toy dataset from scikit-learn (binary targets)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.33, shuffle=False
    )  # split dataset into training and testing sets
    intervals = get_intervals(
        X_train, 10
    )  # get discretization intervals from train set
    X_train = discretize(X_train, intervals)  # discretize train
    X_test = discretize(X_test, intervals)  # discretize test

    # EFC's hyperparameters
    Q = X_test.values.max()
    LAMBDA = 0.5  # pseudocount parameter

    coupling_matrices, h_i_matrices, cutoffs_list = multi_class_fit(
        np.array(X_train, dtype=np.int64), np.array(y_train, dtype=np.int64), Q, LAMBDA
    )  # train model

    y_predicted, y_predicted_proba = multi_class_predict(
        np.array(X_test, dtype=np.int64),
        coupling_matrices,
        h_i_matrices,
        cutoffs_list,
        Q,
        np.unique(y_train),
    )  # test model

    # colect results
    print("Multi-class results")
    print(confusion_matrix(y_test, y_predicted))
    return y_test, y_predicted


if __name__ == "__main__":
    from pprint import pprint

    y_test, y_predicted = single_class_efc()
    pprint(y_test, indent=4)
    pprint(y_predicted, indent=4)
    # _, _ = multi_class_efc()
