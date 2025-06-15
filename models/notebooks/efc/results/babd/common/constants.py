# EFC baseline for experiments
from efc import EnergyBasedFlowClassifier

# int set in train_test_split for reproducibility
RANDOM_STATE = 139

# custom metrics columns name
METRICS_COLUMNS = [
    "Technique",
    "accuracy",
    "f1",
    "f1_micro",
    "f1_macro",
    "precision",
    "recall",
]

# custom confusion matrix labels
LABELS_CM = ["True Negative", "False positive", "False Negative", "True Positive"]

# df size columns name
SIZES_COLUMNS = [
    "Technique",
    "X Size",
    "y Size",
    "X_train Size",
    "X_test Size",
    "y_train Size",
    "y_test Size",
    "y_train Malicious Size",
    "y_train Bening Size",
    "y_test Malicious Size",
    "y_test Bening Size",
]

# EFC set-up
N_BINS = 30  #
CUTOFF_QUANTILE = 0.9  #
PSEUDOCOUNTS = 0.1  #

# efc hyperparameters
EFC_HYPER_PARAMS = {"cutoff_quantile": [0.7, 0.8, 0.9], "n_bins": [10, 20, 30]}

EFC_CLF = EnergyBasedFlowClassifier(
    n_bins=N_BINS, cutoff_quantile=CUTOFF_QUANTILE, pseudocounts=PSEUDOCOUNTS
)
