from kernel import gram_Matrix
from optimization_lib import optimization_restarts_parallel_TNC, optimization_restarts_parallel_CG, gaussian_optimization, optimization_restarts_parallel_LBFGS
from create_data import get_data_set,create_validation_set
from plot_lib import plot_prediction, plot_variance, plot_difference_analytical, plot_both_pred_analy
from posterior import posterior_distribution_f, posterior_distribution_u
from L2error import MSE_error
import numpy as np
from skopt.space import Real

def main():

    #first we need to create a data set
    n_training_points = 25
    noise = [1e-5,1e-5]
    filename = 'gaussian_f_c3.csv'
    x_u, x_f, t_u, t_f, u_train, f_train, raw_data = get_data_set(n_training_points, noise,filename)
    targets_train = np.vstack((u_train, f_train))

    validation_points_base_2 = 10 # 2**10 = 1024
    validation_set = create_validation_set(10, noise, filename)
    
    
    #now we need to optimize the hyperparameters
    #first we need to define the optimization parameters
    dictionary_BFGS_CG_TNC = {'theta_initial': get_initial_values,   #needed for all optimization methods
                      'bounds': ((1e-2, None), (1e-2, None), (1e-1, None),(1e-2, None)), #needed for TNC and L-BFGS-B
                       'gtol': 1e-6}

    #parameters for the Baysian Optimization
    ranges = [Real(1e-1, 3, name='l_x', prior="log-uniform"),
             Real(1e-2, 3, name='sigma_f_sq', prior="uniform"),
             Real(1e-1, 3, name='l_t', prior="log-uniform"),
             Real(1e-1, 4, name='alpha', prior="uniform")]
    bays_opt_dictionary = {'dimensions': ranges,
                           'n_calls': 80,
                           'n_initial_points': 20,
                           'verbose': False,
                           'noise': 1e-4,
                           'n_restarts_optimizer': 6,
                           "n_jobs": -1                       
                           }
    
    #now we can optimize the hyperparameters
    n_restarts = 20
    n_threads = -1
    
    #BFGS_result = optimization_restarts_parallel_LBFGS(gram_Matrix,n_restarts, n_threads, dictionary_BFGS_CG_TNC, X=x_u, Y=x_f, T=t_u, S=t_f, targets=targets_train, noise=noise)
    CJ_result = optimization_restarts_parallel_CG(gram_Matrix,n_restarts, -1, dictionary_BFGS_CG_TNC, X=x_u, Y=x_f, T=t_u, S=t_f, targets=targets_train, noise=noise)
    print("MSE_CJ_result:",MSE_error(validation_set,n_training_points, noise, filename, CJ_result.x))
    #bo_results = gaussian_optimization(bays_opt_dictionary,gram_Matrix,restarts = 10, X=x_u, Y=x_f, T=t_u, S=t_f, targets=targets_train, noise=noise)
    TNC = optimization_restarts_parallel_TNC(gram_Matrix, n_restarts, -1, dictionary_BFGS_CG_TNC, X=x_u, Y=x_f, T=t_u, S=t_f, targets=targets_train, noise=noise)
    print("MSE_TNC_result:",MSE_error(validation_set,n_training_points, noise, filename, TNC.x))
    
    try:
        best_result = min([BFGS_result, CJ_result, bo_results, TNC], key=lambda x: x.fun)
    except:
        pass
    try:
        best_result = min([CJ_result, TNC], key=lambda x: x.fun)
    except:
        best_result = None
   
    print('-------------------------------------best result------------------------------------------')
    print("function value:",best_result.fun,"theta(l_x, sigma_f, l_t, c):",best_result.x)
    print('------------------------------------------------------------------------------------------')

    
    #now we can make predictions
    #first we need to create the test data
    
    n_test_points = 101
    t_test, x_test = np.linspace(0,int(np.max(raw_data[1])),n_test_points).reshape(-1,1), np.linspace(0,1,n_test_points).reshape(-1,1)
    t_star, x_star = np.meshgrid(t_test, x_test)
    t_star, x_star = t_star.reshape(-1,1), x_star.reshape(-1,1)
    
    theta = best_result.x
    #theta = [0.72267089, 0.046254  , 0.50923536, 0.98401197]
    
    mean_u, var_u = posterior_distribution_u(x_u, x_f, t_u, t_f, x_star, t_star, targets_train, noise, theta, gram_Matrix)
    mean_f, var_f = posterior_distribution_f(x_u, x_f, t_u, t_f, x_star, t_star, targets_train, noise, theta, gram_Matrix)
    mean_u = mean_u.reshape(n_test_points,n_test_points)
    mean_f = mean_f.reshape(n_test_points,n_test_points)
    var_u = var_u.reshape(n_test_points,n_test_points)
    var_f = var_f.reshape(n_test_points,n_test_points)
    #now we can plot the results
    plot_prediction(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, mean_u, mean_f, title='Predictive mean and variance', save_path='vibrating_string/plots/predictive_mean.png')
    plot_variance(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, var_u, var_f, title='Predictive mean and variance', save_path='vibrating_string/plots/predictive_variance.png')


    #now we look at the difference between the analytical solution and the predicted mean. because we dont have a function for u and f but only the numerical solution
    # we just calculate the mean for all the points in the grid from the numerical solution 
    x_star = raw_data[0].reshape(-1,1)
    t_star = raw_data[1].reshape(-1,1)
    u_grid = raw_data[2]
    f_grid = raw_data[3]

    mean_u_wolfram_grid, var = posterior_distribution_u(x_u, x_f, t_u, t_f, x_star, t_star, targets_train,noise, theta, gram_Matrix)
    mean_f_wolfram_grid, var = posterior_distribution_f(x_u, x_f, t_u, t_f, x_star, t_star, targets_train,noise, theta, gram_Matrix)
    mean_u_wolfram_grid = mean_u_wolfram_grid.reshape(u_grid.shape)
    mean_f_wolfram_grid = mean_f_wolfram_grid.reshape(u_grid.shape)
    
    plot_difference_analytical(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, mean_u_wolfram_grid, mean_f_wolfram_grid, u_grid, f_grid, title='Difference between analytical solution and predicted mean', save_path='vibrating_string/plots/difference.png')

    print('--------------------MSE-------------------------')
    
    MSE_u, MSE_f = MSE_error(validation_set,n_training_points, noise, filename, theta)
    print(" MSE for u:",MSE_u,"MSE for f" ,MSE_f)

    pass
def get_initial_values():
    """returns the initial values for the hyperparameters
    for the length scales we initialize them randomly as log(l) ~ U(-2.5,1)
    """
    rng = np.random.default_rng()
    theta_initial = np.zeros((4))
    theta_initial[0] = np.exp(rng.uniform(-1.3, 0.3, 1))
    theta_initial[1] = rng.uniform(0, 2, 1)
    theta_initial[2] = np.exp(rng.uniform(-1.3, 0.3, 1))
    theta_initial[3] = rng.uniform(1, 4, 1)
    return theta_initial


if __name__ == '__main__':
    main()