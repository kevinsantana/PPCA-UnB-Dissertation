from constants import EFC_CLF, LAST_TIME_STEP, LAST_TRAIN_TIME_STEP
from shared_functions import (
    calculate_model_score,
    custom_confusion_matrix,
    drop_agg_features,
    get_dataset_size,
    plot_efc_energies,
    run_elliptic_preprocessing_pipeline,
)


def process_labeled_samples(technique:str, fig_folder: str, fig_name: str, include_agg: bool = True):
    if include_agg:
        X_train, X_test, y_train, y_test = run_elliptic_preprocessing_pipeline(last_train_time_step=LAST_TRAIN_TIME_STEP,
                                                                               last_time_step=LAST_TIME_STEP,
                                                                               only_labeled=True)
    else:
        X_train, X_test, y_train, y_test = drop_agg_features(inplace=False)

    clf = EFC_CLF
    clf.fit(X_train, y_train, base_class=0)
    y_pred, y_energies = clf.predict(X_test, return_energies=True)
    plot_efc_energies(clf, y_test, y_energies, fig_folder, fig_name)
    sizes = get_dataset_size(technique=technique, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    metric_dict = calculate_model_score(technique=technique, y_true=y_test.values, y_pred=y_pred)
    confusion_matrix = custom_confusion_matrix(technique=technique, y_test=y_test, y_pred=y_pred)
    return sizes, metric_dict, confusion_matrix
