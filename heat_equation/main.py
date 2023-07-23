
from create_data import get_data_set
from optimization import optimization_restarts_parallel_LBFGS, optimization_restarts_parallel_CG, gaussian_optimization, log_marginal_likelihood, optimization_restarts_parallel_TNC
from gram_matrix import gram_Matrix
from posterior import posterior_distribution_u, posterior_distribution_f
from plot_prediction import plot_prediction, plot_variance, plot_difference_analytical
import numpy as np
from skopt.space import Real
from MSE import mse

def main():

    #first we need to create a data set
    n_training_points = 20
    noise = [1e-4,1e-4]
    u_train, f_train, t_u, x_u, t_f, x_f = get_data_set(n_training_points, noise)
    targets_train = np.vstack((u_train, f_train))
    #now we need to create the gram matrix
    
    #now we need to optimize the hyperparameters
    #first we need to define the optimization parameters
    dictionary_BFGS_CG_TNC = {'theta_initial': get_initial_values,   #needed for all optimization methods
                      'bounds': ((1e-2, 3), (1e-2, 3), (1e-2, 3),(1e-2, None)), #needed for TNC and L-BFGS-B
                       'gtol': 1e-8}

                     
    #parameters for the Baysian Optimization
    ranges = [Real(1e-1, 3, name='l_x', prior="log-uniform"),
             Real(1e-2, 3, name='sigma_f_sq', prior="uniform"),
             Real(1e-1, 3, name='l_t', prior="log-uniform"),
             Real(1e-1, 2, name='alpha', prior="uniform")]
    bays_opt_dictionary = {'dimensions': ranges,
                           'n_calls': 100,
                           'n_initial_points': 20,
                           'random_state':0,
                           'verbose': False,
                           'noise': 1e-7                        
                           }
    
    #now we can optimize the hyperparameters
    n_restarts = 100
    n_threads = 10
    
    #BFGS_result = optimization_restarts_parallel_LBFGS(gram_Matrix,n_restarts, n_threads, dictionary_BFGS_CG_TNC, X=x_u, Y=x_f, T=t_u, S=t_f, targets=targets_train, noise=noise)
    CJ_result = optimization_restarts_parallel_CG(gram_Matrix,n_restarts, -1, dictionary_BFGS_CG_TNC, X=x_u, Y=x_f, T=t_u, S=t_f, targets=targets_train, noise=noise)
    #bo_results = gaussian_optimization(bays_opt_dictionary,gram_Matrix, X=x_u, Y=x_f, T=t_u, S=t_f, targets=targets_train, noise=noise)
    TNC = optimization_restarts_parallel_TNC(gram_Matrix, n_restarts, -1, dictionary_BFGS_CG_TNC, X=x_u, Y=x_f, T=t_u, S=t_f, targets=targets_train, noise=noise)

    try:
        best_result = min([BFGS_result, CJ_result, bo_results, TNC], key=lambda x: x.fun)
    except:
        pass
    try:
        best_result = min([CJ_result, TNC], key=lambda x: x.fun)
    except:
        best_result = TNC
    print('--------------------best result-------------------------')
    print(best_result)
    
    print('--------------------MSE-------------------------')
    #print(mse(x_u, x_f, t_u, t_f,targets_train, 1, best_result.x, noise, gram_Matrix))
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
    plot_prediction(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, mean_u, mean_f, title='Predictive mean and variance', save_path='test.png')
    plot_variance(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, var_u, var_f, title='Predictive mean and variance', save_path='test_variance.png')
    plot_difference_analytical(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, mean_u, mean_f, title='Difference between the analytical solution and the predicted mean', save_path='test_difference.png')
    
    
    
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