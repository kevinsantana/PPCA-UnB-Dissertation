import numpy as np
import sys
sys.path.append('../efc')
import pandas as pd
from classification_functions import *
from generic_discretize import *
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    '''Usage example of Single-class EFC'''
    data = load_breast_cancer(as_frame=True) # load toy dataset from scikit-learn (binary targets)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, shuffle=False) # split dataset into training and testing sets
    intervals = get_intervals(X_train, 10) # get discretization intervals from train set
    X_train = discretize(X_train, intervals) # discretize train
    X_test = discretize(X_test, intervals) # discretize test

    idx_abnormal = np.where(y_train == 1)[0] # find abnormal samples indexes in the training set
    X_train.drop(idx_abnormal, axis=0, inplace=True) # remove abnormal samples from training (EFC trains with only benign instances)
    y_train.drop(idx_abnormal, axis=0, inplace=True) # remove the corresponding abonrmal training targets

    #EFC's hyperparameters
    Q = X_test.values.max()
    LAMBDA = 0.5 # pseudocount parameter

    coupling, h_i, cutoff, _, _ = OneClassFit(np.array(X_train, dtype=np.int64), Q, LAMBDA) # train model
    y_predicted, energies = OneClassPredict(np.array(X_test, dtype=np.int64), coupling, h_i, cutoff, Q) # test model

    # colect results
    print('Single-class results')
    print(confusion_matrix(y_test, y_predicted))


    '''Usage example of Multi-class EFC'''
    data = load_wine(as_frame=True) # load toy dataset from scikit-learn (binary targets)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, shuffle=False) # split dataset into training and testing sets
    intervals = get_intervals(X_train, 10) # get discretization intervals from train set
    X_train = discretize(X_train, intervals) # discretize train
    X_test = discretize(X_test, intervals) # discretize test

    #EFC's hyperparameters
    Q = X_test.values.max()
    LAMBDA = 0.5 # pseudocount parameter

    coupling_matrices, h_i_matrices, cutoffs_list = MultiClassFit(np.array(X_train, dtype=np.int64), np.array(y_train, dtype=np.int64), Q, LAMBDA) # train model
    y_predicted, y_predicted_proba = MultiClassPredict(np.array(X_test, dtype=np.int64), coupling_matrices, h_i_matrices, cutoffs_list, Q, np.unique(y_train)) # test model

    # colect results
    print('Multi-class results')
    print(confusion_matrix(y_test, y_predicted))


if __name__ == "__main__":
    main()
