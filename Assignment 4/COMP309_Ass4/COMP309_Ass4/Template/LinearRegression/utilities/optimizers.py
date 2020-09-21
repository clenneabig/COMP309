# -*- coding: utf-8 -*-

"""
Optimizers for optimizing the performance metrics include:

1. Batch Gradient Descent
""" 

import numpy as np

# from utilities.losses import compute_loss
from utilities.losses import compute_loss


def gradient_descent(y, x, theta, max_iters, alpha, metric_type):
    """
    Batch Gradient Descent
    :param y:               ground truth
    :param x:               input data (feature matrix)
    :param theta:           model parameters (w and b)
    :param max_iters:       max iterations
    :param alpha:           step size
    :param metric_type:     metric type
    :return: thetas         all tracked updated model parameters
             losses         all tracked losses during the learning course
    """
    losses = []
    thetas = []
    num_of_samples = len(x)
    for i in range(max_iters):
        # This is for MSE loss only
        gradient = -2 * x.T.dot(y - x.dot(theta)) / num_of_samples
        theta = theta - alpha * gradient
        loss = compute_loss(y, x, theta, metric_type)

        # Track losses and thetas
        thetas.append(theta)
        losses.append(loss)

        print("BGD({bi}/{ti}): loss={l}, w={w}, b={b}".format(
            bi = i, ti = max_iters - 1, l = loss, w = theta[0], b = theta[1]))
    return thetas, losses


def mini_batch_gradient_descent(y, x, theta, max_iters, alpha, metric_type, mini_batch_size):
    """
    Mini Batch Gradient Descent
    :param y:               ground truth
    :param x:               input data (feature matrix)
    :param theta:           model parameters (w and b)
    :param max_iters:       max iterations
    :param alpha:           step size
    :param metric_type:     metric type
    :param mini_batch_size: mini batch size
    :return: thetas         all tracked updated model parameters
             losses         all tracked losses during the learning course
    """
    losses = []
    thetas = []
    # Please refer to the function "gradient_descent" to implement the mini-batch gradient descent here
    return thetas, losses


def pso(y, x, theta, max_iters, pop_size, metric_type):
    """
    Particle Swarm Optimization
    :param y:               train labels
    :param x:               train data
    :param theta:           model parameters
    :param max_iters:       max iterations
    :param pop_size:        population size
    :param metric_type:     metric type (MSE, RMSE, R2, MAE)
    :return: best_thetas    all tracked best model parameters for each generation
             losses         all tracked losses of the best model in each generation
    """
    # Init settings
    w = 0.729844  # Inertia weight to prevent velocities becoming too large
    c_p = 1.496180  # Scaling co-efficient on the social component
    c_g = 1.496180  # Scaling co-efficient on the cognitive component

    terminate = False
    g_best = theta

    lower_bound = -100
    upper_bound = 100

    velocity = []
    thetas = []
    p_best = []

    # Track results
    best_thetas = []
    losses = []

    # initialization
    for i in range(pop_size):
        theta = np.random.uniform(lower_bound, upper_bound, len(theta))
        thetas.append(theta)
        p_best.append(theta)
        if compute_loss(y, x, theta, metric_type) < compute_loss(y, x, g_best, metric_type):
            g_best = theta.copy()
        velocity.append(
            np.random.uniform(-np.abs(upper_bound - lower_bound), np.abs(upper_bound - lower_bound), len(theta)))

    # Evolution
    count = 0
    while not terminate:
        for i in range(pop_size):
            rand_p = np.random.uniform(0, 1, size = len(theta))
            rand_g = np.random.uniform(0, 1, size = len(theta))
            velocity[i] = w * velocity[i] + c_p * rand_p * (p_best[i] - thetas[i]) + c_g * rand_g * (g_best - thetas[i])
            thetas[i] = thetas[i] + velocity[i]
            if compute_loss(y, x, thetas[i], metric_type) < compute_loss(y, x, p_best[i], metric_type):
                p_best[i] = thetas[i]
                if compute_loss(y, x, p_best[i], metric_type) < compute_loss(y, x, g_best, metric_type):
                    g_best = p_best[i]
        best_thetas.append(g_best)
        current_loss = compute_loss(y, x, g_best, metric_type)
        losses.append(current_loss)

        print("PSO({bi}/{ti}): loss={l}, w={w}, b={b}".format(
            bi = count, ti = max_iters - 1, l = current_loss, w = g_best[0], b = g_best[1]))
        count += 1
        if count >= max_iters:
            terminate = True
    return best_thetas, losses
