import os
from pprint import pprint

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from efc import EnergyBasedFlowClassifier
from shared_functions import (
    drop_agg_features,
    run_elliptic_preprocessing_pipeline,
    train_test_from_splitted,
    train_test_from_x_y,
    get_dataset_size,
    plot_efc_energies,
    calculate_model_score,
    custom_confusion_matrix
)


from constants import N_BINS, CUTOFF_QUANTILE, PSEUDOCOUNTS, LAST_TRAIN_TIME_STEP, LAST_TIME_STEP


def make_feature_selection_with_k_best(k: int,
                                       fig_folder: str,
                                       fig_name: str,
                                       X: pd.DataFrame = None,
                                       y: pd.Series = None
    ) -> None:
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    try:
        if not all([X, y]):
            X, y = run_elliptic_preprocessing_pipeline(
            last_train_time_step=LAST_TRAIN_TIME_STEP,
            last_time_step=LAST_TIME_STEP,
            only_x_y=True
        )
    except ValueError:
        pass

    pprint(f"X shape: {X.shape}", indent=4)
    X_new = SelectKBest(f_classif, k=k).fit_transform(X, y)
    pprint(f"X_new shape: {X_new.shape}", indent=4)
    X_train, X_test, y_train, y_test = train_test_from_x_y(X_new, y)
    sizes = get_dataset_size(f"Feature Selection Score Function f_classif k={k}", X_train, X_test, y_train, y_test)
    clf = EnergyBasedFlowClassifier(n_bins=N_BINS, cutoff_quantile=CUTOFF_QUANTILE, pseudocounts=PSEUDOCOUNTS)
    clf.fit(X_train, y_train)
    y_pred, y_energies = clf.predict(X_test, return_energies=True)
    plot_efc_energies(clf, y_test, y_energies, fig_folder, fig_name)
    model_score = calculate_model_score(technique=f"Feature Selection Score Function f_classif k={k}", y_true=y_test.values, y_pred=y_pred)
    confusion_matrix_values = custom_confusion_matrix(technique=f"Feature Selection Score Function f_classif k={k}",
                                                      y_test=y_test,
                                                      y_pred=y_pred)
    return sizes, model_score, confusion_matrix_values


def make_feature_selection_with_k_best_no_agg_features(k: int,
                                                       fig_folder: str,
                                                       fig_name: str) -> None:
    X_train, X_test, y_train, y_test = drop_agg_features()
    X, y = train_test_from_splitted(X_train, y_train, X_test, y_test, return_df=False)
    sizes, model_score, confusion_matrix_values = make_feature_selection_with_k_best(k,
                                                                                     fig_folder,
                                                                                     fig_name,
                                                                                     X=X,
                                                                                     y=y)
    return sizes, model_score, confusion_matrix_values
