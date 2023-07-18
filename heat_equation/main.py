import numpy as np
from create_data import get_data_set
from optimization import optimization_restarts_parallel, log_marginal_likelihood, log_marginal_likelihood_to_optimize
from gram_matrix import gram_Matrix
from posterior import posterior_distribution_u, posterior_distribution_f
from plot_prediction import plot_prediction, plot_variance
from scipy.optimize import minimize

def main():

    #first we need to create a data set
    n_training_points = 20
    noise = [1e-3,1e-3]
    u_train, f_train, t_u, x_u, t_f, x_f = get_data_set(n_training_points, noise)
    targets_train = np.vstack((u_train, f_train))
    #now we need to create the gram matrix
    
    #now we need to optimize the hyperparameters
    #first we need to define the optimization parameters
    opt_dictionary = {'theta_initial': get_initial_values,
                      'bounds': ((1e-2, None), (1e-2, None), (1e-2, None),(1e-2, None)),
                       'gtol': 1e-8}
    
    #now we can optimize the hyperparameters
    n_restarts = 1000
    n_threads = -1
   
    best_result = optimization_restarts_parallel(gram_Matrix,n_restarts, n_threads, opt_dictionary, X=x_u, Y=x_f, T=t_u, S=t_f, targets=targets_train, noise=noise)
    print(best_result)
    #now we can make predictions
    #first we need to create the test data
    n_test_points = 100
    t_test, x_test = np.linspace(0,1,n_test_points).reshape(-1,1), np.linspace(0,1,n_test_points).reshape(-1,1)
    t_star, x_star = np.meshgrid(t_test, x_test)
    t_star, x_star = t_star.reshape(-1,1), x_star.reshape(-1,1)
    
    theta = best_result.x
    
    
    mean_u, var_u = posterior_distribution_u(x_u, x_f, t_u, t_f, x_star, t_star, targets_train, noise, theta, gram_Matrix)
    mean_f, var_f = posterior_distribution_f(x_u, x_f, t_u, t_f, x_star, t_star, targets_train, noise, theta, gram_Matrix)
    mean_u = mean_u.reshape(n_test_points,n_test_points)
    mean_f = mean_f.reshape(n_test_points,n_test_points)
    var_u = var_u.reshape(n_test_points,n_test_points)
    var_f = var_f.reshape(n_test_points,n_test_points)
    #now we can plot the results
    plot_prediction(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, mean_u, mean_f, title='Predictive mean and variance', save_path='heat_equation/test.png')
    plot_variance(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, var_u, var_f, title='Predictive mean and variance', save_path='heat_equation/test_variance.png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass
def get_initial_values():
    """returns the initial values for the hyperparameters
    for the length scales we initialize them randomly as log(l) ~ U(-2.5,1)
    """
    rng = np.random.default_rng()
    theta_initial = np.zeros((4))
    theta_initial[0] = np.exp(rng.uniform(-1.3, 0.3, 1))
    theta_initial[1] = rng.uniform(0, 1, 1)
    theta_initial[2] = np.exp(rng.uniform(-1.3, 0.3, 1))
    theta_initial[3] = rng.uniform(0, 2, 1)
    return theta_initial


if __name__ == '__main__':
    main()