#This file contains the log marginal likelihood function and the optimization function.
from scipy.optimize import minimize
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from jax import jit, grad, vmap, jacfwd
import jax.numpy as jnp
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import numpy as np

def log_marginal_likelihood(Kernel:callable, X, Y, T, S, targets, noise:list, params:list) -> float:
    """calculates the log marginal likelihood as given in Rasmussen p. 19. 
    params = l_x, sigma_f_sq, l_t, alpha  
    noise = noise_u, noise_f
    """
    
    K = Kernel(X, Y, T, S, params, noise)  
    
    L = jnp.linalg.cholesky(K + 1e-6 * jnp.zeros(K.shape)) #add some jitter for stability
    #if jnp.isnan(L).any():
     #   return jnp.array(1e6)

    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, targets))
    mll = 1/2 * jnp.dot(targets.T,alpha) +0.5*jnp.sum(jnp.log(jnp.diagonal(L))) + len(X)/2 * jnp.log(2*jnp.pi)
    
    return jnp.squeeze(mll)

def log_marginal_likelihood_to_optimize(Kernel:callable,  X, Y, T, S, targets, noise):
    """returns the log marginal likelihood as a function of the hyperparameters, so it can be used in the optimization function"""
    def function_to_optimize(hyperparams):
        mll = log_marginal_likelihood(Kernel, X, Y, T, S, targets,noise, hyperparams)
        return mll
    return function_to_optimize

def optimization_restarts_parallel_LBFGS(Kernel: callable, n_restarts: int, n_threads: int,opt_dictionary: dict, X, Y, T, S, targets, noise) -> dict:
    """
    performs the optimization of the hyperparameters in parallel and returns the best result. This uses 
    n_restarts: number of restarts of the optimization
    n_threads: number of threads to use for the parallelization
    opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
    """
    def single_optimization_run(Kernel, X, Y, T, S, targets, noise):
        """performs a single optimization run with random initialization of the hyperparameters"""
        theta_initial = opt_dictionary['theta_initial']()
        
        res = minimize(log_marginal_likelihood_to_optimize(Kernel, X, Y, T, S, targets,noise), x0=theta_initial,
                       method='L-BFGS-B', bounds=opt_dictionary['bounds'],
                       options={'gtol': opt_dictionary['gtol']})
        return res

    
    results = Parallel(n_jobs=n_threads)(delayed(single_optimization_run)(Kernel, X, Y, T, S, targets,noise) for _ in tqdm(range(n_restarts)))

    valid_results = [res for res in results if res.success]
    best_result = min(valid_results, key=lambda x: x.fun)
    print(best_result)
    return best_result
###############################################################

def grad_log_marginal_likelihood(Kernel:callable, X, Y, T, S, targets, noise:list):
    return jacfwd(log_marginal_likelihood_to_optimize(Kernel, X, Y, T, S, targets, noise))


def optimization_restarts_parallel_CG(Kernel: callable, n_restarts: int, n_threads: int,opt_dictionary: dict, X, Y, T, S, targets, noise) -> dict:
    """
    performs the optimization of the hyperparameters in parallel and returns the best result.
    n_restarts: number of restarts of the optimization
    n_threads: number of threads to use for the parallelization
    opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
    """
    def single_optimization_run(Kernel, X, Y, T, S, targets, noise):
        """performs a single optimization run with random initialization of the hyperparameters"""
        theta_initial = opt_dictionary['theta_initial']()
        
        res = minimize(log_marginal_likelihood_to_optimize(Kernel, X, Y, T, S, targets,noise), x0=theta_initial,
                       method='CG')# jac=jit(grad_log_marginal_likelihood(Kernel, X, Y, T, S, targets,noise)),bounds=opt_dictionary['bounds']
        return res

    
    results = Parallel(n_jobs=n_threads)(delayed(single_optimization_run)(Kernel, X, Y, T, S, targets,noise) for _ in tqdm(range(n_restarts)))
    #all positive parameters
    results = [res for res in results if  np.all(res.x > 0) and res.success]
    best_result = min(results, key=lambda x: x.fun)
    print(best_result)
    return best_result

from skopt import gp_minimize, forest_minimize
from skopt.space import Real

def gaussian_optimization(dictionary: dict, Kernel, X, Y, T, S, targets, noise):
 
    
    def objective_function(params):
        l_x, sigma_f_sq, l_t, alpha = params
        return log_marginal_likelihood(Kernel, X, Y, T, S, targets, noise, [l_x, sigma_f_sq, l_t, alpha]).item()  # we minimize -log_likelihood

    result = gp_minimize(objective_function,  
                        **dictionary)
    print(result.x, result.fun)
    return result

def optimization_restarts_parallel_TNC(Kernel: callable, n_restarts: int, n_threads: int,opt_dictionary: dict, X, Y, T, S, targets, noise) -> dict:
    """
    performs the optimization of the hyperparameters in parallel and returns the best result.
    n_restarts: number of restarts of the optimization
    n_threads: number of threads to use for the parallelization
    opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
    """
    def single_optimization_run(Kernel, X, Y, T, S, targets, noise):
        """performs a single optimization run with random initialization of the hyperparameters"""
        theta_initial = opt_dictionary['theta_initial']()
        res = minimize(log_marginal_likelihood_to_optimize(Kernel, X, Y, T, S, targets,noise), x0=theta_initial,
                       method='TNC', jac=jit(grad_log_marginal_likelihood(Kernel, X, Y, T, S, targets,noise)),bounds=opt_dictionary['bounds'])
        return res

    
    results = Parallel(n_jobs=n_threads)(delayed(single_optimization_run)(Kernel, X, Y, T, S, targets,noise) for _ in tqdm(range(n_restarts)))
    #all positive parameters
    results = [res for res in results if  np.all(res.x > 0) and res.success]
    best_result = min(results, key=lambda x: x.fun)
    print(best_result)
    return best_result