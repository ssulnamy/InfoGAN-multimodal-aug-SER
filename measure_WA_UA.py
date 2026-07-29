#-*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score, recall_score


'''
list_y_ture : reference (label)
list_y_pred : predicted value
note        : do not consider "label imbalance"
'''
def weighted_accuracy(list_y_true, list_y_pred):

    assert(len(list_y_true) == len(list_y_pred))
           
    y_true = np.array(list_y_true)
    y_pred = np.array(list_y_pred)
           
    return accuracy_score(y_true=y_true, y_pred=y_pred)
    
    
'''
list_y_ture : reference (label)
list_y_pred : predicted value
note        : compute accuracy for each class; then, average the computed accurcies
              consider "label imbalance"
''' 
def unweighted_accuracy(list_y_true, list_y_pred):
    
    assert(len(list_y_true) == len(list_y_pred))
           
    y_true = np.array(list_y_true).reshape(-1)
    y_pred = np.array(list_y_pred)
    
    w = np.ones(y_true.shape[0])
    for idx, i in enumerate(np.bincount(y_true)):
        w[y_true == idx] = float(1/i)
    
    return accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=w)

def uar_score(labels: np.ndarray,
              predictions: np.ndarray):
    """
    Computes the unweighted average recall for the specified true labels and predictions.
    
    The unweighted average recall is simply the average recall for each class without any weighting.
    
    Parameters
    ----------
    labels: numpy.ndarray
        A one-dimensional numpy array containing the true labels of instances
    predictions
        A one-dimensional numpy array containing the predicted labels of instances

    Returns
    -------
    float
        The unweighted average recall for the specified true labels and predictions
    """
    return recall_score(labels.reshape(-1), predictions.reshape(-1), average="macro")