import numpy as np
from functools import partial
from scipy.optimize import minimize

def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    n = len(y)
    pred = X@theta
    margin = -y*pred
    log_loss = np.logaddexp(0, margin).sum()
    reg = l2_param*(theta@theta)
    return log_loss/n + reg
    
def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    objective_function = partial(objective_function, X=X, y=y, l2_param=l2_param)
    
    n_features = X.shape[1]
    theta_0 = np.zeros(n_features)
    theta = minimize(objective_function, theta_0).x
    return theta
        