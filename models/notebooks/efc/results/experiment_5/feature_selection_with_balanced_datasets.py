import pandas as pd
from results.experiment_3.unbalanced_techniques import smote
from results.experiment_4.feature_selection_selectkbest_f_classif import (
    make_feature_selection_with_k_best,
)


def smote_with_feature_selection(
    technique: str, fig_folder: str, fig_name: str, k: int, only_labeled: bool = True
):
    X, y = make_feature_selection_with_k_best(
        k=k, fig_folder=fig_folder, fig_name=fig_name, return_x_y=True
    )
    X_df = pd.DataFrame(X)
    return smote(
        technique=technique,
        fig_folder=fig_folder,
        fig_name=fig_name,
        only_labeled=only_labeled,
        X=X_df,
        y=y,
        test_complete_dataset=True,
        is_resampled=True,
    )
