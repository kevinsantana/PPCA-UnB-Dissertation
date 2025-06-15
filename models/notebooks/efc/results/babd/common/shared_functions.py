# Model performance functions

import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from babd.common.constants import LABELS_CM
from efc import EnergyBasedFlowClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def custom_confusion_matrix(
    technique: str, y_test: np.ndarray, y_pred: np.ndarray
) -> Dict[str, int]:
    """Calculates and formats a confusion matrix.

    Computes the confusion matrix for the given true labels (`y_test`) and predicted labels (`y_pred`), and returns it as a dictionary
    with descriptive labels. It assumes binary classification (0 and 1) and the 'labels' argument of confusion matrix is explicitly set to [1, 0].

    Args:
        y_test (np.ndarray): True labels (0 or 1).
        y_pred (np.ndarray): Predicted labels (0 or 1).
        technique (str): A string describing the technique used for the predictions (e.g., "EFC", "Baseline"). This is used as a label in the output dictionary.

    Returns:
        dict: A dictionary representing the confusion matrix, with keys:
            - "Technique":  The input `technique` string.
            - "True Negative":  Number of true negatives.
            - "False positive": Number of false positives.
            - "False Negative": Number of false negatives.
            - "True Positive": Number of true positives.

    Raises:
        ValueError: If `y_test` or `y_pred` contain values other than 0 or 1.  This function is designed for binary classification.
    """
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    cm = np.reshape(cm, -1).tolist()
    return {"Technique": technique} | {label: val for val, label in zip(cm, LABELS_CM)}


def get_dataset_size(
    technique: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, int]:
    """
    Calculates and returns the sizes of the training and testing sets.

    Computes the sizes of the overall dataset, the training and testing sets for both features (X) and labels (y), and the
    number of malicious and benign samples in the training and testing sets.

    Args:
        y_train (np.ndarray): Training labels (0 or 1).
        y_test (np.ndarray): Testing labels (0 or 1).
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.

    Returns:
        dict: A dictionary containing the dataset sizes, with the following keys:
            - "X Size": Total number of samples in X.
            - "y Size": Total number of samples in y.
            - "X_train Size": Number of samples in X_train.
            - "X_test Size": Number of samples in X_test.
            - "y_train Size": Number of samples in y_train.
            - "y_test Size": Number of samples in y_test.
            - "y_train Malicious Size": Number of malicious samples (label 1) in y_train.
            - "y_train Bening Size": Number of benign samples (label 0) in y_train.
            - "y_test Malicious Size": Number of malicious samples (label 1) in y_test.
            - "y_test Bening Size": Number of benign samples (label 0) in y_test.

    Raises:
        TypeError: If any of the inputs are not NumPy arrays.
        ValueError: If y_train or y_test contains elements that are not either 0 or 1. This assumes binary labels.
    """
    return {
        "Technique": technique,
        "X Size": len(X_train) + len(X_test),
        "y Size": len(y_train) + len(y_test),
        "X_train Size": len(X_train),
        "X_test Size": len(X_test),
        "y_train Size": len(y_train),
        "y_test Size": len(y_test),
        "y_train Malicious Size": len(np.where(y_train == 1)[0]),
        "y_train Bening Size": len(np.where(y_train == 0)[0]),
        "y_test Malicious Size": len(np.where(y_test == 1)[0]),
        "y_test Bening Size": len(np.where(y_test == 0)[0]),
    }


def calculate_model_score(
    technique: str,
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, int]:
    """
    Calculates common classification metrics.

    Computes accuracy, weighted F1-score, micro F1-score, macro F1-score, weighted precision, and weighted recall for the given true and predicted labels.

    Args:
        y_true (Union[np.ndarray, pd.Series]): True labels.
        y_pred (Union[np.ndarray, pd.Series]): Predicted labels.

    Returns:
        dict: A dictionary containing the calculated metrics:
            - "accuracy": Accuracy score.
            - "f1": Weighted F1-score.
            - "f1_micro": Micro F1-score.
            - "f1_macro": Macro F1-score.
            - "precision": Weighted precision.
            - "recall": Weighted recall.

    Raises:
        ValueError: If the input labels are invalid (e.g., contain values outside the expected range for the classification task).
        TypeError: if input arguments have an unexpected type or if labels are not in an expected format.
    """
    return {
        "Technique": technique,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
    }


def plot_efc_energies(
    y_test: pd.Series,
    y_energies: pd.Series,
    fig_name: str,
    fig_folder: str,
    show_plot: bool = True,
    clf: Optional[EnergyBasedFlowClassifier] = None,
    cutoff: Optional[pd.Series] = None,
) -> None:
    """Plots the energy distributions for benign and malicious transactions.

    Generates a histogram visualizing the energy distributions of benign and malicious transactions as predicted by an EnergyBasedFlowClassifier (EFC).
    It highlights the decision cutoff used by the classifier.  The plot is saved to a file.

    Args:
        clf (EnergyBasedFlowClassifier): The trained EFC classifier.
        y_test (pd.Series): True labels (0 for benign, 1 for malicious).
        y_energies (pd.Series): Energies assigned by the EFC to each transaction.
        fig_folder (str): Path to the directory where the plot will be saved.
        fig_name (str): Filename for the saved plot (e.g., "energy_distribution.png").

    Returns:
        None. The function saves the plot to a file.

    Raises:
        FileNotFoundError: If the `fig_folder` directory does not exist and cannot be created.
        ValueError: If `y_test` or `y_energies` have inconsistent shapes, or if `y_test` contains values other than 0 or 1.
        OSError: If there are any issues writing the plot file to disk (e.g., permissions issues).
    """
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # ploting energies
    benign = np.where(y_test == 0)[0]
    malicious = np.where(y_test == 1)[0]

    benign_energies = y_energies[benign]
    malicious_energies = y_energies[malicious]
    if clf:
        plt.clf()
        cutoff = clf.estimators_[0].cutoff_
    else:
        cutoff = cutoff

    bins = np.histogram(y_energies, bins=60)[1]

    plt.hist(
        malicious_energies,
        bins,
        facecolor="#006680",
        alpha=0.7,
        ec="white",
        linewidth=0.3,
        label="malicious",
    )
    plt.hist(
        benign_energies,
        bins,
        facecolor="#b3b3b3",
        alpha=0.7,
        ec="white",
        linewidth=0.3,
        label="benign",
    )
    plt.axvline(cutoff, color="r", linestyle="dashed", linewidth=1)
    plt.legend()

    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.savefig(f"{fig_folder}/{fig_name}")
    if show_plot:
        plt.show()
