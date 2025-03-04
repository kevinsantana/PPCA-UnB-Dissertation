#################################
### Dataset related functions ###
#################################

import os
import sys
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from efc import EnergyBasedFlowClassifier
from results.common.constants import LAST_TIME_STEP, LAST_TRAIN_TIME_STEP, LABELS_CM


ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
ONLY_LABELED = True


def combine_dataframes(df_classes: pd.DataFrame, df_features: pd.DataFrame, only_labeled: bool = True) -> pd.DataFrame:
    """
    Combines features and class labels based on transaction IDs.

    Merges the features DataFrame (`df_features`) with the class labels DataFrame (`df_classes`) using a left merge 
    based on the transaction IDs. Optionally filters out transactions with unknown labels (class 2).

    Args:
        df_classes (pd.DataFrame): DataFrame containing transaction IDs and their corresponding class labels.
            Must have columns "txId" and "class".
        df_features (pd.DataFrame): DataFrame containing transaction IDs and their features. Must have a column "id".
        only_labeled (bool, optional): Whether to filter out transactions with unknown labels (class 2). Defaults to True.

    Returns:
        pd.DataFrame: Combined DataFrame with features and class labels. The "txId" column from `df_classes`
            is dropped.

    Raises:
        KeyError: If "txId" or "class" columns are missing in `df_classes`, or if "id" column is missing in `df_features`.
        MergeError: If the merge operation fails due to unexpected data or column inconsistencies.
    """
    df_combined = pd.merge(
        df_features, df_classes, left_on="id", right_on="txId", how="left"
    )
    if only_labeled == True:
        df_combined = df_combined[df_combined["class"] != 2].reset_index(drop=True)
    df_combined.drop(columns=["txId"], inplace=True)

    return df_combined


def rename_classes(df_classes: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Renames the class labels in the provided DataFrame.

    Replaces string class labels ("1", "2", "unknown") with numerical labels (1, 0, 2) in the "class" column.
    Specifically, it maps "1" (illicit) to 1, "2" (licit) to 0, and "unknown" to 2.  Modifies the DataFrame in place.

    Args:
        df_classes (pd.DataFrame):  DataFrame containing the class labels.  Must have a column named "class".

    Returns:
        pd.DataFrame: The modified DataFrame with renamed class labels.

    Raises:
        KeyError: If the "class" column is not present in the input DataFrame.
        TypeError: If the "class" column contains data types that cannot be compared to strings (e.g., numeric types).
    """
    if inplace:
        df_classes.replace({"class": {"1": 1, "2": 0, "unknown": 2}}, inplace=True)
        return df_classes
    else:
        new_df = df_classes.replace({"class": {"1": 1, "2": 0, "unknown": 2}}, inplace=True)
        return new_df


def rename_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the feature columns in the provided DataFrame.

    Replaces the default numeric feature names with more descriptive names. The new names follow the pattern:
    "id", "time_step", "trans_feat_0" to "trans_feat_92", and "agg_feat_0" to "agg_feat_71".

    Args:
        df_features (pd.DataFrame): DataFrame containing the features.  It is assumed to have 167 columns
            (1 ID, 1 timestamp, 93 transaction features, and 72 aggregated features).

    Returns:
        pd.DataFrame: The DataFrame with renamed feature columns.

    Raises:
        IndexError: If the input DataFrame does not have the expected number of columns (167).  This suggests the
            data is not in the expected format for the Elliptic dataset.
    """
    df_features.columns = (
        ["id", "time_step"]
        + [f"trans_feat_{i}" for i in range(93)]
        + [f"agg_feat_{i}" for i in range(72)]
    )
    return df_features


def import_elliptic_data_from_csvs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Imports the Elliptic dataset from CSV files.

    Reads the Elliptic dataset from three CSV files: "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv", and
    "elliptic_txs_features.csv". Assumes the files are located in the "datasets/elliptic" subdirectory relative to the
    current working directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three DataFrames:
            - df_classes: DataFrame containing transaction IDs and their corresponding class labels.
            - df_edges: DataFrame containing the transaction graph edge list.
            - df_features: DataFrame containing transaction IDs and their features.

    Raises:
        FileNotFoundError: If any of the required CSV files are not found in the specified location.
        pd.errors.ParserError: If there is an error parsing any of the CSV files (e.g., incorrect format, invalid data).
        OSError: If there are any other operating system errors during file access.
    """
    df_classes = pd.read_csv(
        os.path.join(ROOT_DIR, "datasets/elliptic/elliptic_txs_classes.csv")
    )
    df_edges = pd.read_csv(
        os.path.join(ROOT_DIR, "datasets/elliptic/elliptic_txs_edgelist.csv")
    )
    df_features = pd.read_csv(
        os.path.join(ROOT_DIR, "datasets/elliptic/elliptic_txs_features.csv"),
        header=None,
    )
    return df_classes, df_edges, df_features


def setup_train_test_idx(
    X: pd.DataFrame, last_train_time_step: int, last_time_step: int,
    aggregated_timestamp_column: Optional[str]="time_step"
) -> Dict[str, pd.Index]:
    """
    Creates train/test indices based on the time_step column.

    This function generates training and testing indices for a temporal split of the data.  It uses the 
    `aggregated_timestamp_column` to determine the split point.  The data is split such that all transactions with a 
    timestamp less than or equal to `last_train_time_step` are in the training set, and all transactions with a timestamp 
    greater than `last_train_time_step` and less than or equal to `last_time_step` are in the testing set.

    Args:
        X (pd.DataFrame): The input DataFrame containing the data and the timestamp column.
        last_train_time_step (int): The last time step to include in the training set.
        last_time_step (int): The last time step to include in the testing set.
        aggregated_timestamp_column (str, optional): The name of the column containing the timestamps. 
            Defaults to "time_step".

    Returns:
        Dict[str, pd.Index]: A dictionary containing the training and testing indices.  The keys are "train" and "test",
            and the values are Pandas Index objects.

    Raises:
        KeyError: If the `aggregated_timestamp_column` is not found in the DataFrame `X`.
        TypeError: If the values in the `aggregated_timestamp_column` are not comparable to integers.  The timestamp column 
            should contain numerical data representing discrete time steps.
        ValueError: If `last_train_time_step` is negative or greater than `last_time_step`.
    """
    split_timesteps = {}

    split_timesteps["train"] = list(range(last_train_time_step + 1))
    split_timesteps["test"] = list(range(last_train_time_step + 1, last_time_step + 1))

    train_test_idx = {}
    train_test_idx["train"] = X[
        X[aggregated_timestamp_column].isin(split_timesteps["train"])
    ].index
    train_test_idx["test"] = X[
        X[aggregated_timestamp_column].isin(split_timesteps["test"])
    ].index

    return train_test_idx


def train_test_split(X: pd.DataFrame,
                     y: pd.Series,
                     train_test_idx: Dict[str, pd.Index]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets based on pre-calculated indices.

    Uses the provided `train_test_idx` dictionary to extract the training and testing subsets of the input data `X` and labels `y`.

    Args:
        X (pd.DataFrame): The input DataFrame containing the features.
        y (pd.Series): The input Series containing the labels.
        train_test_idx (Dict[str, pd.Index]): A dictionary containing the training and testing indices.  The keys must be 
            "train" and "test", and the values should be Pandas Index objects.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training and testing data and labels:
            - X_train_df: Training data as a DataFrame.
            - X_test_df: Testing data as a DataFrame.
            - y_train: Training labels as a Series.
            - y_test: Testing labels as a Series.

    Raises:
        KeyError: If the `train_test_idx` dictionary is missing the "train" or "test" keys.
        IndexingError: If the indices in `train_test_idx` are out of bounds for `X` or `y`.  This can occur if the indices 
            were generated from a different DataFrame or if the DataFrames have been modified since the indices were created.
        TypeError: If `X` or `y` are not the expected data types (DataFrame and Series, respectively).
    """
    X_train_df = X.loc[train_test_idx["train"]]
    X_test_df = X.loc[train_test_idx["test"]]

    y_train = y.loc[train_test_idx["train"]]
    y_test = y.loc[train_test_idx["test"]]
    return X_train_df, X_test_df, y_train, y_test


def load_elliptic_data(only_labeled: bool = True, drop_node_id: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and preprocesses the Elliptic dataset.

    Imports the data from CSV files, renames classes and features, combines the DataFrames, and prepares the data for 
    machine learning by separating features (X) and labels (y).

    Args:
        only_labeled (bool, optional): Whether to include only transactions with known labels (class 0 or 1). 
            If False, transactions with unknown labels (class 2) are also included. Defaults to True.
        drop_node_id (bool, optional): Whether to drop the transaction ID ("id") column from the features. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the features (X) and labels (y):
            - X: DataFrame of features. If `drop_node_id` is True, the "id" column is dropped.
            - y: Series of class labels (0 for licit, 1 for illicit, 2 for unknown).

    Raises:
        FileNotFoundError: If any of the Elliptic dataset CSV files are not found.
        pd.errors.ParserError: If there's an error parsing the CSV files.
        KeyError: If any expected columns ("id", "class", "txId") are missing in the DataFrames.
        IndexError: If the `df_features` DataFrame doesn't have the expected 167 columns after reading from CSV.
        MergeError: If the merge operation in `combine_dataframes` fails.  This can occur if there are inconsistencies between the data in 
            `df_classes` and `df_features`.


    """
    df_classes, df_edges, df_features = import_elliptic_data_from_csvs()
    df_features = rename_features(df_features)
    df_classes = rename_classes(df_classes)
    df_combined = combine_dataframes(df_classes, df_features, only_labeled)

    if drop_node_id == True:
        X = df_combined.drop(columns=["id", "class"])
    else:
        X = df_combined.drop(columns="class")

    y = df_combined["class"]
    return X, y


def run_elliptic_preprocessing_pipeline(
    last_train_time_step: int,
    last_time_step: int,
    only_labeled: bool = True, 
    drop_node_id: bool = True,
    only_x_y: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Runs the complete Elliptic dataset preprocessing pipeline.

    Loads the data, renames classes and features, combines DataFrames, performs a temporal train/test split, and returns the 
    training and testing sets.

    Args:
        last_train_time_step (int): The last time step to include in the training set.
        last_time_step (int): The last time step to be included in the test set.
        only_labeled (bool, optional): Whether to include only transactions with known labels (0 or 1). Defaults to True.
        drop_node_id (bool, optional): Whether to drop the "id" column (transaction ID). Defaults to True.
        only_x_y (bool, optional): If True, returns only the combined X and y before splitting. If False (default), performs the train/test split.

    Returns:
        tuple: If `only_x_y` is False, returns a tuple containing the training and testing data and labels:
            - X_train_df (pd.DataFrame): Training features.
            - X_test_df (pd.DataFrame): Testing features.
            - y_train (pd.Series): Training labels.
            - y_test (pd.Series): Testing labels.

            If `only_x_y` is True, returns a tuple containing the combined X and y DataFrames
            - X (pd.DataFrame): Features.
            - y (pd.Series): Labels.


    Raises:
        FileNotFoundError: If the Elliptic dataset CSV files are not found.
        pd.errors.ParserError: If there is an error parsing the CSV files.
        KeyError: If expected columns are missing in the DataFrames.
        IndexError: If `df_features` doesn't have the expected number of columns.
        MergeError: If the merge operation fails.
        KeyError: If the time_step column is not available in the input dataframe
        TypeError: If the time_step column cannot be compared to integers.
        ValueError: if last_train_time_step is negative or greater than last_time_step.
    """
    X, y = load_elliptic_data(only_labeled, drop_node_id)
    if only_x_y:
        return X, y

    train_test_idx = setup_train_test_idx(X, last_train_time_step, last_time_step)
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, train_test_idx)

    return X_train_df, X_test_df, y_train, y_test


def train_test_from_splitted(X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             return_df: bool = False
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """
    Combines split data back into a single DataFrame or numpy arrays.

    This function takes the training and testing sets (X_train, y_train, X_test, y_test) and combines them back into a single 
    DataFrame if `return_df` is True, or into X and y numpy arrays if `return_df` is False. This is useful if you need to 
    modify the split data (e.g., feature dropping) and then want to work with the full dataset again.

    Args:
        X_train (pd.DataFrame): Training features DataFrame.
        y_train (pd.Series): Training labels Series.
        X_test (pd.DataFrame): Testing features DataFrame.
        y_test (pd.Series): Testing labels Series.
        return_df (bool, optional): If True, returns a single combined DataFrame. If False (default), returns X and y numpy arrays.

    Returns:
        Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]: 
            - If `return_df` is True, returns a single combined DataFrame.
            - If `return_df` is False, returns a tuple containing:
                - X (pd.DataFrame): Combined features DataFrame.
                - y (pd.Series): Combined labels Series.


    Raises:
        TypeError: If the input data types are not as expected (DataFrames for X_train and X_test, Series for y_train and y_test).
        ValueError: If the input DataFrames have incompatible shapes or indices, preventing concatenation. For example, if the number of features in X_train and X_test doesn't match.
    """
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    df = pd.concat([df_train, df_test])
    X = df.drop(['class'], axis=1)
    y = df['class']

    if return_df:
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
        df_new = pd.concat([df_train, df_test])
        return df_new

    return X, y  


def recreate_original_df() -> pd.DataFrame:
    """
    Recreates the original Elliptic dataset DataFrame.

    This function uses the preprocessing pipeline with predefined parameters to reconstruct the original Elliptic DataFrame
    as if it was just loaded and preprocessed, including renaming features and classes, and combining dataframes,
    but *without* performing a train/test split.  It uses a fixed `last_time_step` of 49, 
    `last_train_time_step` of 34, and includes only labeled transactions (`only_labeled=True`).

    Returns:
        pd.DataFrame: The combined and preprocessed Elliptic DataFrame, similar to what you'd get after loading and preprocessing 
            but before splitting into train/test sets.

    Raises:
        FileNotFoundError: If any of the Elliptic dataset CSV files are not found.
        pd.errors.ParserError: If there is an error parsing the CSV files.
        KeyError: If expected columns are missing in the DataFrames.
        IndexError: If the features DataFrame doesn't have the expected number of columns.
        MergeError: If the merge operation in `combine_dataframes` fails.
    """
    X_train, X_test, y_train, y_test = run_elliptic_preprocessing_pipeline(last_train_time_step=LAST_TRAIN_TIME_STEP,
                                                                             last_time_step=LAST_TIME_STEP,
                                                                             only_labeled=ONLY_LABELED)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    df = pd.concat([df_train, df_test])
    return df


def train_test_from_x_y(X: Union[pd.DataFrame, np.ndarray],
                        y: Union[pd.Series, np.ndarray]
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Performs a stratified train-test split.

    This function uses scikit-learn's `train_test_split` to split the data `X` and labels `y` into training and testing sets.
    The split is stratified based on `y` to maintain class proportions in both sets.  It uses a fixed `random_state` for reproducibility
    and shuffles the data before splitting. The `test_size` is set to 0.3 (30% for testing).

    Args:
        X (Union[pd.DataFrame, np.ndarray]): The input features. Can be a Pandas DataFrame or a NumPy array.
        y (Union[pd.Series, np.ndarray]): The input labels. Can be a Pandas Series or a NumPy array.

    Returns:
        Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
            A tuple containing the training and testing data and labels:
            - X_train: Training features.  The type will be the same as the input `X`.
            - X_test: Testing features. The type will be the same as the input `X`.
            - y_train: Training labels. The type will be the same as the input `y`.
            - y_test: Testing labels. The type will be the same as the input `y`.

    Raises:
        ValueError: If the input data `X` or labels `y` are not valid data types (DataFrame, Series, or array).
        ValueError: If there are issues with stratification (e.g., not enough samples of a particular class).
    """
    X_train_df, X_test_df, y_train, y_test = sklearn_train_test_split(X, y, random_state=139, stratify=y, shuffle=True, test_size=0.3)
    return X_train_df, X_test_df, y_train, y_test


def is_dataframe_non_empty(df: pd.DataFrame) -> bool:
  """
  Checks if a Pandas DataFrame is non-empty.

  Args:
      df: The Pandas DataFrame to check.

  Returns:
      True if the DataFrame is non-empty, False otherwise.
  """
  if df.empty:
    return False
  else:
    return True


def drop_agg_features(X_train: pd.DataFrame = None,
                      X_test: pd.DataFrame = None,
                      y_train: pd.DataFrame = None,
                      y_test: pd.DataFrame = None,
                      inplace: bool = False
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Drops aggregated features from training and testing DataFrames.

    Removes the columns corresponding to aggregated features ('agg_feat_0' to 'agg_feat_71') from the input DataFrames.
    This implementation has issues with inplace modification and unnecessary operations on y_train and y_test.

    Args:
        X_train (pd.DataFrame): Training features DataFrame.
        X_test (pd.DataFrame): Testing features DataFrame.
        y_train (pd.DataFrame): Training labels (should not contain aggregated features).
        y_test (pd.DataFrame): Testing labels (should not contain aggregated features).
        inplace (bool, optional): If True, supposedly modifies DataFrames in place, but the inner drop calls always use inplace=True,
            leading to incorrect behavior. Defaults to False.

    Returns:
        tuple: A tuple of the modified DataFrames (or None if inplace=True due to the flaw):
            - X_train_new (pd.DataFrame or None): Modified training features or None.
            - X_test_new (pd.DataFrame or None): Modified testing features or None.
            - y_train_new (pd.DataFrame or None): Training labels (potentially modified incorrectly or None).
            - y_test_new (pd.DataFrame or None): Testing labels (potentially modified incorrectly or None).

    Raises:
        KeyError: If any of the aggregated feature columns ('agg_feat_0' to 'agg_feat_71') are not found
            in X_train or X_test, or if they are unexpectedly found in y_train or y_test.
    """
    if not all([X_train, X_test, y_train, y_test]):  # check if any of the dataframes are empty
        X_train, X_test, y_train, y_test = run_elliptic_preprocessing_pipeline(last_train_time_step=LAST_TRAIN_TIME_STEP,
                                                                               last_time_step=LAST_TIME_STEP,
                                                                               only_labeled=ONLY_LABELED)
        inplace = False
    if not inplace:
        X_train_new = X_train.drop(columns=[f"agg_feat_{i}" for i in range(72)], axis=1, inplace=False)
        X_test_new = X_test.drop(columns=[f"agg_feat_{i}" for i in range(72)], axis=1, inplace=False)
        y_train_new = y_train.drop(columns=[f"agg_feat_{i}" for i in range(72)], axis=1, inplace=False)
        y_test_new = y_test.drop(columns=[f"agg_feat_{i}" for i in range(72)], axis=1, inplace=False)
        return X_train_new, X_test_new, y_train_new, y_test_new
    else:
        X_train.drop(columns=[f"agg_feat_{i}" for i in range(72)], axis=1, inplace=True)
        X_test.drop(columns=[f"agg_feat_{i}" for i in range(72)], axis=1, inplace=True)
        y_train.drop(columns=[f"agg_feat_{i}" for i in range(72)], axis=1, inplace=True)
        y_test.drop(columns=[f"agg_feat_{i}" for i in range(72)], axis=1, inplace=True)


def save_to_csv(df: pd.DataFrame, file_name: str, sep=',', encoding='utf-8', index=False, header=True) -> None:
    df.to_csv(file_name, sep=sep, encoding=encoding, index=index, header=header)



###################################
### Model performance functions ###
###################################


def custom_confusion_matrix(technique: str, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
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
    return {'Technique': technique} | {label: val for val, label in zip(cm, LABELS_CM)}


def get_dataset_size(technique: str,
                    X_train: np.ndarray,
                    X_test: np.ndarray,
                    y_train: np.ndarray,
                    y_test: np.ndarray
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


def calculate_model_score(technique: str,
                          y_true: Union[np.ndarray, pd.Series],
                          y_pred: Union[np.ndarray, pd.Series]
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



def plot_efc_energies(y_test: pd.Series,
                      y_energies: pd.Series,
                      results_folder: str,
                      fig_name: str,
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
        results_folder (str): Path to the directory where the plot will be saved.
        fig_name (str): Filename for the saved plot (e.g., "energy_distribution.png").

    Returns:
        None. The function saves the plot to a file.

    Raises:
        FileNotFoundError: If the `results_folder` directory does not exist and cannot be created.
        ValueError: If `y_test` or `y_energies` have inconsistent shapes, or if `y_test` contains values other than 0 or 1.
        OSError: If there are any issues writing the plot file to disk (e.g., permissions issues).
    """
    # make experiments results dir if not exists
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # ploting energies
    benign = np.where(y_test == 0)[0]
    malicious = np.where(y_test == 1)[0]

    benign_energies = y_energies[benign]
    malicious_energies = y_energies[malicious]
    if clf:
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
    plt.savefig(f'{results_folder}/{fig_name}')

