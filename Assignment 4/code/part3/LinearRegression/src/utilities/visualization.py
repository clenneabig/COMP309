# -*- coding: utf-8 -*-
 
"""
Visualization functions
"""

import matplotlib.pyplot as plt
import numpy as np

# Visualize the training course
from utilities.losses import compute_loss


def compute_z_loss(y, x, thetas):
    """
    Compute z-axis values
    :param y:            train labels
    :param x:            train data
    :param thetas:       model parameters
    :return: z_losses    value (loss) for z-axis
    """
    thetas = np.array(thetas)
    w = thetas[:, 0].reshape(thetas[:, 0].shape[0], )
    b = thetas[:, 1].reshape(thetas[:, 1].shape[0], )
    z_losses = np.zeros((len(w), len(b)))
    for ind_row, row in enumerate(w):
        for ind_col, col in enumerate(b):
            theta = np.array([row, col])
            z_losses[ind_row, ind_col] = compute_loss(y, x, theta, "MSE")
    return z_losses


def predict(x, thetas):
    """
    Predict function
    :param x:               test data
    :param thetas:          trained model parameters
    :return:                prediced labels
    """
    return x.dot(thetas)


def visualize_train(train_data_full, train_labels, train_data, thetas, losses, niter):
    """
    Visualize Function for Training Results
    :param train_data_full:   the train data set (full) with labels and data
    :param thetas:            model parameters
    :param losses:            all tracked losses
    :param niter:             completed training iterations
    :return: fig1              the figure for line fitting on training data
             fig2              learning curve in terms of error
             fig3              gradient variation visualization
    """
    fig1, ax1 = plt.subplots()
    ax1.scatter(train_data_full["Weight"], train_data_full["Height"], color = 'blue')

    # De-standarize
    train_mean = train_data_full["Weight"].mean()
    train_std = train_data_full["Weight"].std()
    train_data_for_plot = train_mean + train_data["Weight"] * train_std

    ax1.plot(train_data_for_plot, predict(train_data, thetas[niter - 1]), color = 'red', linewidth = 2)
    ax1.set_xlabel("Height")
    ax1.set_ylabel("Weight")

    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(losses)), losses, color = 'blue', linewidth = 2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MSE")

    fig3, ax3 = plt.subplots()
    np_gradient_ws = np.array(thetas)

    w = np.linspace(min(np_gradient_ws[:, 0]), max(np_gradient_ws[:, 0]), len(np_gradient_ws[:, 0]))
    b = np.linspace(min(np_gradient_ws[:, 1]), max(np_gradient_ws[:, 1]), len(np_gradient_ws[:, 1]))
    x, y = np.meshgrid(w, b)
    z = compute_z_loss(train_labels, train_data, np.stack((w,b)).T)
    cp = ax3.contourf(x, y, z, cmap = plt.cm.jet)
    fig3.colorbar(cp, ax = ax3)
    ax3.plot(3.54794951, 66.63949115837143, color = 'red', marker = '*', markersize = 20)
    if niter > 0:
        thetas_to_plot = np_gradient_ws[:niter]
    ax3.plot(thetas_to_plot[:, 0], thetas_to_plot[:, 1], marker = 'o', color = 'w', markersize = 10)
    ax3.set_xlabel(r'$w$')
    ax3.set_ylabel(r'$b$')
    return fig1, fig2, fig3


def visualize_test(test_data_full, test_data, thetas):
    """
    Visualize Test for Testing Results
    :param test_data_full:          the test data set (full) with labels and data
    :param thetas:                  model parameters
    :return: fig
    """
    fig, ax = plt.subplots()
    ax.scatter(test_data_full["Weight"], test_data_full["Height"], color='blue')
    ax.plot(test_data_full["Weight"], predict(test_data, thetas[-1]), color='red', linewidth=2)
    return fig
