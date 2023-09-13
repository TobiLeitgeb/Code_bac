from create_data import get_data_set
from kernel import gram_Matrix
from posterior import posterior_distribution_f, posterior_distribution_u
import numpy as np


def MSE_error(validation_set,n_training_points, noise, filename, theta):
    """calculates the MSE error between the analytical solution and the predicted mean
        calculates the mean for all the points from the validation set with the found theta and the training data
        !!!very important to use the same training set as was used for the optimization!!!
    """
    x_u, x_f, t_u, t_f, u_train, f_train, raw_data = get_data_set(n_training_points, noise, filename)
    targets_train = np.vstack((u_train,f_train))

    x_star_u = validation_set[0].reshape(-1,1)
    x_star_f = validation_set[1].reshape(-1,1)
    t_star_u = validation_set[2].reshape(-1,1)
    t_star_f = validation_set[3].reshape(-1,1)
    u_grid = validation_set[4]
    f_grid = validation_set[5]
    mean_u, var = posterior_distribution_u(x_u, x_f, t_u, t_f, x_star_u, t_star_u, targets_train, noise, theta, gram_Matrix)
    mean_f, var = posterior_distribution_f(x_u, x_f, t_u, t_f, x_star_f, t_star_f, targets_train, noise, theta, gram_Matrix)
    mean_u = mean_u.reshape(u_grid.shape)
    mean_f = mean_f.reshape(f_grid.shape)

    MSE_u = np.mean(np.abs(mean_u - u_grid))
    MSE_f = np.mean(np.abs(mean_f - f_grid))
    return MSE_u, MSE_f