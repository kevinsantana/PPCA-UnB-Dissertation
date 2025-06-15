import os

import pandas as pd
from babd.common.constants import RANDOM_STATE
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def get_project_root() -> str:
    """
    Dynamically determines the project root directory.

    This function traverses up the directory tree from the current file's location
    until it finds the directory containing the 'models' directory, which is
    assumed to be a unique marker of the project root.

    Returns:
        str: The absolute path to the project root directory.

    Raises:
        RuntimeError: If the project root directory cannot be found.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.sep:  # Stop at the root directory
        if "models" in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise RuntimeError("Could not find project root directory.")


def run_babd_preprocessing_pipeline(only_x_y: bool = False):
    X, y = load_babd_data()
    if only_x_y:
        return X, y

    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y)

    return X_train_df, X_test_df, y_train, y_test


def load_babd_data():
    df = import_babd_data_from_csv()
    X = df.drop(columns=["addaccount", "SW", "label"])
    y = df["label"]
    return X, y


def import_babd_data_from_csv() -> pd.DataFrame:
    project_root = get_project_root()
    df = pd.read_csv(
        os.path.join(
            project_root,
            "models/notebooks/efc/datasets/BABD/BABD-13.csv",
        )
    )
    return df


def train_test_split(X, y):
    X_train, X_test, y_train, y_test = sklearn_train_test_split(
        X, y, random_state=RANDOM_STATE, stratify=y, shuffle=True, test_size=0.3
    )
    return X_train, X_test, y_train, y_test
