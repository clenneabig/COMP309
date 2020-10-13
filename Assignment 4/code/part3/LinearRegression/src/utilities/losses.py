# -*- coding: utf-8 -*-

"""
A function used to compute for the loss

"""  

import numpy as np


def compute_loss(y, x, theta, metric_type):
    """
    Compute the loss of given data with respect to the ground truth
      y            ground truth
      x            input data (feature matrix)
      theta        model parameters (w and b)
      metric_type  metric type seletor, e.g., "MSE" indicates the Mean Squared Error.
    """
    if metric_type.upper() == "MSE":
        return np.mean(np.power(x.dot(theta) - y, 2))
    elif metric_type.upper() == "RMSE":
        return np.sqrt(np.mean(np.power(x.dot(theta) - y, 2)))
    elif metric_type.upper() == "R2":
        return - (1 - np.mean(np.power(x.dot(theta) - y, 2)) / np.mean(np.power(y - np.mean(y), 2)))
    elif metric_type.upper() == "MAE":
        return np.mean(np.abs(y - x.dot(theta)))


