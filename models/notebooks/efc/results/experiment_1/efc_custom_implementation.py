from math import ceil

import numpy as np
from constants import EFC_CLF, LAST_TIME_STEP, LAST_TRAIN_TIME_STEP
from efc_python.classification_functions import one_class_fit, one_class_predict
from efc_python.generic_discretize import discretize, get_intervals
from shared_functions import (
    calculate_model_score,
    custom_confusion_matrix,
    get_dataset_size,
    plot_efc_energies,
    run_elliptic_preprocessing_pipeline,
)


def efc_custom(technique: str, fig_folder: str, fig_name: str):
    X_train, X_test, y_train, y_test = run_elliptic_preprocessing_pipeline(
        last_train_time_step=LAST_TRAIN_TIME_STEP,
        last_time_step=LAST_TIME_STEP,
        only_labeled=True,
    )
    intervals = get_intervals(
        X_train, 10
    )  # get discretization intervals from train set
    X_train_df = discretize(X_train, intervals)  # discretize train
    X_test_df = discretize(X_test, intervals)  # discretize test
    idx_abnormal = np.where(y_train == 1)[
        0
    ]  # find abnormal samples indexes in the training set
    X_train_df.drop(
        idx_abnormal, axis=0, inplace=True
    )  # remove abnormal samples from training (EFC trains with only benign instances)
    y_train.drop(
        idx_abnormal, axis=0, inplace=True
    )  # remove the corresponding abonrmal training targets
    Q = int(X_test.values.max())  # EFC's hyperparameters
    LAMBDA = 0.5  # pseudocount parameter
    coupling, h_i, cutoff, _, _ = one_class_fit(
        np.array(X_train_df), Q, LAMBDA
    )  # train model
    y_pred, y_energies = one_class_predict(
        np.array(X_test_df), coupling, h_i, cutoff, Q
    )  # test model
    plot_efc_energies(y_test, y_energies, fig_folder, fig_name, cutoff=cutoff)
    sizes = get_dataset_size(
        technique=technique,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    metric_dict = calculate_model_score(
        technique=technique, y_true=y_test.values, y_pred=y_pred
    )
    confusion_matrix = custom_confusion_matrix(
        technique=technique, y_test=y_test, y_pred=y_pred
    )
    return sizes, metric_dict, confusion_matrix


def efc_with_percentage_labeled(
    technique: str, fig_folder: str, fig_name: str, percentage: int
):
    X_train, X_test, y_train, y_test = run_elliptic_preprocessing_pipeline(
        last_train_time_step=LAST_TRAIN_TIME_STEP,
        last_time_step=LAST_TIME_STEP,
        only_labeled=True,
    )
    if not isinstance(percentage, int) or percentage < 0 or percentage > 100:
        raise ValueError("Invalid percentage. Should be an integer between 0 and 100.")

    interval = (100 - percentage) / 100
    slice_size = ceil(len(indices_illicit) * interval)
    intervals = get_intervals(
        X_train, 10
    )  # get discretization intervals from train set
    X_train = discretize(X_train, intervals)  # discretize train
    X_test = discretize(X_test, intervals)  # discretize test
    indices_illicit = np.where(y_train == 1)[0]
    drop_indices_illicit = indices_illicit[:slice_size]
    # retrieve idxs abnormals and choose %percentage of them
    # drop random labeled indices
    X_train.drop(
        drop_indices_illicit, axis=0, inplace=True
    )  # remove abnormal samples from training (EFC trains with only benign instances)
    y_train.drop(
        drop_indices_illicit, axis=0, inplace=True
    )  # remove the corresponding abonrmal training targets
    # EFC's hyperparameters
    Q = np.int64(X_test.values.max())
    LAMBDA = 0.5  # pseudocount parameter
    coupling, h_i, cutoff, _, _ = one_class_fit(
        np.array(X_train), Q, LAMBDA
    )  # train model
    y_pred, y_energies = one_class_predict(
        np.array(X_test), coupling, h_i, cutoff, Q
    )  # test model
    plot_efc_energies(y_test, y_energies, fig_folder, fig_name, cutoff=cutoff)
    sizes = get_dataset_size(
        technique=technique,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    metric_dict = calculate_model_score(
        technique=technique, y_true=y_test.values, y_pred=y_pred
    )
    confusion_matrix = custom_confusion_matrix(
        technique=technique, y_test=y_test, y_pred=y_pred
    )
    return sizes, metric_dict, confusion_matrix
