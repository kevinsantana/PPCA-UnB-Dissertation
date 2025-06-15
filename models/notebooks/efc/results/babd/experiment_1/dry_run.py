from typing import Optional

import pandas as pd
from babd.common.constants import EFC_CLF
from babd.common.shared_functions import (
    calculate_model_score,
    custom_confusion_matrix,
    get_dataset_size,
    plot_efc_energies,
)


def dry_run_babd(
    fig_folder: str,
    fig_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    technique: Optional[str] = None,
):
    clf = EFC_CLF
    clf.fit(X_train, y_train)
    y_pred, y_energies = clf.predict(X_test, return_energies=True)
    plot_efc_energies(
        clf=clf,
        y_test=y_test,
        y_energies=y_energies,
        fig_name=fig_name,
        fig_folder=fig_folder,
    )
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
