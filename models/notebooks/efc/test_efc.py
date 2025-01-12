import numpy as np

from efc_python.classification_functions import (
    one_class_fit,
    one_class_predict,
)
from research_aml_elliptic.src.experiments.general_functions.elliptic_data_preprocessing import run_elliptic_preprocessing_pipeline


# import Elliptic data set and set variables
last_time_step = 49
last_train_time_step = 34
only_labeled = True


X_train, X_test, y_train, y_test = run_elliptic_preprocessing_pipeline(last_train_time_step=last_train_time_step,
                                                                             last_time_step=last_time_step,
                                                                             only_labeled=only_labeled)

# EFC's hyperparameters
Q = np.int64(X_test.values.max())
LAMBDA = 0.5  # pseudocount parameter

# train model
coupling, h_i, cutoff, _, _ = one_class_fit(np.array(X_train), Q, LAMBDA)