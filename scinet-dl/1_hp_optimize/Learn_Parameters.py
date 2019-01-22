#
# Advanced neural networks: hyperparameter optimization.
# Given at SciNet, 2 October 2017, by Erik Spence.
#
# This file, Learn_parameters.py, contains the code used for lecture
# 1.  It contains the parts needed to optimize the hyperparameters in
# a neural network.  Note that you must supply your own machine
# learning function to optimize.
#

#######################################################################


"""
Learn_Parameters.py contains the code needed to implement a
generic machine-learning hyperparameter optimization using Gaussian
Processes.  This code is heavily inspired by
https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

"""

#######################################################################


import numpy as np
import numpy.random as npr

import scipy.optimize as sco
import sklearn.gaussian_process as gp
from scipy.stats import norm as ssn

# This code is heavily inspired by 
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py


#########################################################


def expected_improvement(x, model, f_train, n_params = 1):

    """
    Given a combination of hyperparameters, x, the previously
    calculated set of trained NN scores, f_train, and a Gaussian
    Processes model, calculate EI(x), the expected improvement.
        Inputs:

    - x: list of integers, the combination of hyperparameters whose EI
      we are calculating.
    
    - model: the Gaussian Processes model which is used to calculate EI.

    - f_train: list of floats, calculated NN scores for previously
          evaluated hyperparameter combinations.

    - n_params: int, the number of parameters being tested.  Default: 1.

    Outputs:

    - Float: the negative of the expected improvement.

    """

    # Initialize.
    new_x = x.reshape(-1, n_params)
    best_f = np.max(f_train)

    # Get the mean and standard deviation of the Gaussian for this
    # value.
    mu, sigma = model.predict(new_x, return_std = True)

    # in case sigma equals zero
    z = (mu - best_f) / sigma
    result = (mu - best_f) * ssn.cdf(z) + sigma * ssn.pdf(z)

    # We flip the sign because we're trying to minimize this
    # function.
    return -1 * result


#########################################################


def get_next_param(model, f_train, bounds, n_restarts = 25):

    """
    This function is used to calculate the next parameter to evaluate.
    Given the previously calculated set of trained NN scores, f_train,
    and a Gaussian Processes model, find the hyperpameter combination
    which maximizes the expected improvement.  

    Inputs:

    - model: the Gaussian Processes model which is used to calculate EI.

    - f_train: list of floats, calculated NN scores for previously
          evaluated hyperparameter combinations.

    - bounds: 2D array, where each row in the array is the range of
      possible values for a given parameter.

    - n_restart: int, the number of times we'll try to search through
      the parameter space to find the next hyperparameter combination.
      Default: 25.

    Outputs:

    - list of integers: the next hyperparameter combination to try.

    """

    # Initialize.
    best_x = None
    best_value = 1
    n_params = bounds.shape[0]

    # Randomly choose some points from which to start our search
    # through parameter space.
    for point in npr.uniform(bounds[:, 0],
                             bounds[:, 1], 
                             size = (n_restarts, n_params)):

        # And for each one minimize the expected improvement function
        # above.
        res = sco.minimize(fun = expected_improvement,
                           x0 = point.reshape(1, -1),
                           bounds = bounds,
                           method = 'L-BFGS-B',
                           args = (model, f_train, n_params))

        # If we've beaten the previous best value, then update best_f
        # and best_x.
        if res.fun < best_value:
            best_value = res.fun
            best_x = res.x

    # Return the current best HP combination.
    return [int(round(i)) for i in best_x]


#########################################################


def call_nn(next_x, nn_func, best_f, best_x, X_train, f_train):

    """
    This function just calls the external program which calculates the
    score for the given hyperparameter combination, next_x, given the
    name of the function to call, the current best_f value, the list
    of previously tested HP combinations, and the previously
    calculated set of trained NN scores, f_train.

    Inputs:

    - next_x: the next HP combination to test.
    
    - nn_func: the name of the function to call to calculate the score
      for this HP combination.

    - best_f: float, the current best score.

    - best_x: list, the HP combination associated with best_f.

    - X_train: list of the previously-test HP combinations.

    - f_train: list of floats, calculated NN scores for previously
          evaluated hyperparameter combinations.

    Outputs:

    - tuple, the current best_f and best_x

    """

    # Train the NN with these parameters.
    next_f = nn_func(next_x)

    # Add them to the collection.
    X_train.append(next_x)
    f_train.append(next_f)

    # Check to see if we have improvement.
    if (next_f > best_f):
        best_f = next_f
        best_x = next_x

    # Return the current best.
    return best_f, best_x
    

#########################################################


def gp_search(num_iters, nn_func, bounds, num_presamples = 5):

    """
    This function is the work horse that runs the actual search for the
    best hyperparameter (HP) combination.

    Inputs:

    - num_iters: int, the number to iterations to search for the best
      HP combination.
    
    - nn_func: the name of the function to call to calculate the score
      for this HP combination.

    - bounds: 2D array, where each row in the array is the range of
      possible values for a given parameter.

    - num_presamples: int, the number to random samples to put into
      the Gaussian Process model before starting the search.

    Outputs:

    - tuple, the current best_f and best_x

    """

    # Initialize.
    X_train = []
    f_train = []
    n_params = bounds.shape[0]
    best_f = 0.0
    best_x = []

    # Randomly populate the model with points from the HP space.
    print 'Determining the starting points.'
    for params in npr.uniform(bounds[:, 0], bounds[:, 1], 
                               (num_presamples, n_params)):

        # Make the random values integers, and calculate the scores.
        next_x = [int(round(i)) for i in params]
        best_f, best_x = call_nn(next_x, nn_func, best_f, best_x,
                                 X_train, f_train)


    # Initialize the Gaussian Process model.
    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(kernel = kernel,
                                        normalize_y = True)

    # Start actively searching for the best HP combination.
    print 'Starting optimization.'
    for n in range(num_iters):

        # Fit the model to the current set of parameters and scores.
        model.fit(np.array(X_train), np.array(f_train))

        # Calculate the next parameter to test.
        next_x = get_next_param(model, np.array(f_train), 
                                bounds, n_restarts = 100)

        # Calculate the score for this latest HP combination.
        best_f, best_x = call_nn(next_x, nn_func, best_f, best_x,
                                 X_train, f_train)

    # Return the best combination.
    return best_x, best_f


#########################################################

