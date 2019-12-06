# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 08:37:05 2019

@author: robin
"""

import os
import time
import datetime
import argparse
from contextlib import contextmanager


import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

from rdkit import Chem
from rdkit.Chem import AllChem

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from sklearn.metrics import confusion_matrix

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    print("{}...".format(label))
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))

def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError("File '{}' does not exists.".format(path))
    return pd.read_csv(path, delimiter=delimiter)

def create_fingerprints(chemical_compounds):
    """
    Create a learning matrix `X` with (Morgan) fingerprints
    from the `chemical_compounds` molecular structures.

    Parameters
    ----------
    chemical_compounds: array [n_chem, 1] or list [n_chem,]
        chemical_compounds[i] is a string describing the ith chemical
        compound.

    Return
    ------
    X: array [n_chem, 124]
        Generated (Morgan) fingerprints for each chemical compound, which
        represent presence or absence of substructures.
    """
    n_chem = chemical_compounds.shape[0]

    nBits = 124
    X = np.zeros((n_chem, nBits))

    for i in range(n_chem):
        m = Chem.MolFromSmiles(chemical_compounds[i])
        X[i,:] = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=124)

    return X

def make_submission(y_predicted, auc_predicted, file_name="submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predicted: array [n_predictions, 1]
        if `y_predict[i]` is the prediction
        for chemical compound `i` (or indexes[i] if given).
    auc_predicted: float [1]
        The estimated ROCAUC of y_predicted.
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Creating default indexes if not given
    if indexes is None:
        indexes = np.arange(len(y_predicted))+1

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"Chem_ID","Prediction"\n')
        handle.write('Chem_{:d},{}\n'.format(0,auc_predicted))

        for n,idx in enumerate(indexes):

            if np.isnan(y_predicted[n]):
                raise ValueError('The prediction cannot be NaN')
            line = 'Chem_{:d},{}\n'.format(idx, y_predicted[n])
            handle.write(line)
    return file_name

def get_irrelevant_variables_indexes(X_LS, y_LS, error):
    ecarts = np.zeros(len(X_LS[0]))
    for j in range(len(X_LS[0])):
        c0 = 0
        c1 = 0
        p0 = 0
        p1 = 0
        for i in range(len(X_LS[:,0])):
            if X_LS[:,j][i] == 0:
                c0 += 1.
                if y_LS[i] == 1:
                    p0 += 1.
            if X_LS[:,j][i] == 1:
                c1 += 1.
                if y_LS[i] == 1:
                    p1 += 1.
        p0 = p0 / (c0 + c1)
        p1 = p1 / (c0 + c1)
        ecarts[j] = abs(p0 - p1)
    #m = np.mean(ecarts)
    return np.where(ecarts < error)[0]

def binaryToDecimal(l):
    n = 0
    i = 0
    for e in np.flip(l):
        if e == 1:
            n += 2**i
        i += 1
    return n

def plot_confusion_matrix(y_LS_testing, y_predict):
    conf_matr = confusion_matrix(y_LS_testing, y_predict)
    ax= plt.subplot()
    sn.heatmap(conf_matr, annot=True, fmt='g', ax = ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    #df_cm = pd.DataFrame(conf_matr, index = [i for i in "01"],columns = [i for i in "01"])

    plt.figure(figsize = (2,2))
        

if __name__ == "__main__":
    """LS = load_from_csv("data/training_set.csv")
    TS = load_from_csv("data/test_set.csv")

    # Create fingerprint features and output
    with measure_time("Creating fingerprint"):
        X_LS = create_fingerprints(LS["SMILES"].values)
    y_LS = LS["ACTIVE"].values
    
    with measure_time("Creating fingerprint"):
        X_TS = create_fingerprints(TS["SMILES"].values)
    #irr_indexes = get_irrelevant_variables_indexes(X_LS, y_LS, 0.00)
    
    #X_LS = SelectKBest(chi2, k=120).fit_transform(X_LS, y_LS)
    
    X_LS_learning, X_LS_testing, y_LS_learning, y_LS_testing = train_test_split(X_LS, y_LS, test_size=0.33, random_state=42)
    
    #model = DecisionTreeClassifier()
    #model = KNeighborsClassifier(21)
    #model = SVC(kernel='rbf',gamma='scale',probability=True)
    model = MLPClassifier(hidden_layer_sizes=(1000))
    model.fit(X_LS_learning, y_LS_learning)
    #proba que la valeur de retour soit 1
    y_predict = model.predict_proba(X_LS_testing)[:,1]
    auc_predicted = roc_auc_score(y_LS_testing, y_predict)
    print(auc_predicted)
    model.fit(X_LS, y_LS)
    
    y_predict = model.predict_proba(X_TS)[:,1]
    make_submission(y_predict, auc_predicted, 'submission_MLPClassifier')"""
    
    """#https://medium.com/@mohtedibf/in-depth-parameter-tuning-for-knn-4c0de485baf6
    neighbors = range(1,50)
    train_results = []
    test_results = []
    for n in neighbors:
       model = KNeighborsClassifier(n_neighbors=n)
       model.fit(X_LS_learning, y_LS_learning)
       train_pred = model.predict_proba(X_LS_learning)[:,1]
       roc_auc = roc_auc_score(y_LS_learning, train_pred)
       train_results.append(roc_auc)
       y_pred = model.predict_proba(X_LS_testing)[:,1]
       roc_auc = roc_auc_score(y_LS_testing, y_pred)
       test_results.append(roc_auc)
       print(n)
    
    line1 = plt.plot(neighbors, train_results, 'b', label="Train AUC")
    line2 = plt.plot(neighbors, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('n_neighbors')
    plt.show()"""
    
    