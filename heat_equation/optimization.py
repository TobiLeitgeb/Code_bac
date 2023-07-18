#This file contains the log marginal likelihood function and the optimization function.
from scipy.optimize import minimize
from jax import jit, grad, vmap
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
    if jnp.linalg.eigvals(K+ 1e-6 * jnp.zeros(K.shape)).min() < 0:
        #print('nan')
        return np.inf
    
    L = jnp.linalg.cholesky(K + 1e-6 * jnp.zeros(K.shape)) #add some jitter for stability
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, targets))
    mll = 1/2 * jnp.dot(targets.T,alpha) +0.5*jnp.sum(jnp.log(jnp.diagonal(L))) + len(X)/2 * jnp.log(2*jnp.pi)
    return jnp.squeeze(mll)

def log_marginal_likelihood_to_optimize(Kernel:callable,  X, Y, T, S, targets, noise):
    """returns the log marginal likelihood as a function of the hyperparameters, so it can be used in the optimization function"""
    def function_to_optimize(hyperparams):
        mll = log_marginal_likelihood(Kernel, X, Y, T, S, targets,noise, hyperparams)
        return mll
    return function_to_optimize

def optimization_restarts_parallel(Kernel: callable, n_restarts: int, n_threads: int,opt_dictionary: dict, X, Y, T, S, targets, noise) -> dict:
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
                       method='L-BFGS-B', bounds=opt_dictionary['bounds'],
                       options={'gtol': opt_dictionary['gtol']})
        return res

    
    results = Parallel(n_jobs=n_threads)(delayed(single_optimization_run)(Kernel, X, Y, T, S, targets,noise) for _ in tqdm(range(n_restarts)))

    
    best_result = min(results, key=lambda x: x.fun)

    return best_result
###############################################################

def optimization_restarts_parallel(Kernel: callable, n_restarts: int, n_threads: int,opt_dictionary: dict, X, Y, T, S, targets, noise) -> dict:
    """
    performs the optimization of the hyperparameters in parallel and returns the best result.
    n_restarts: number of restarts of the optimization
    n_threads: number of threads to use for the parallelization
    opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
    """
    def single_optimization_run(Kernel, X, Y, T, S, targets, noise):
        """performs a single optimization run with random initialization of the hyperparameters"""
        #theta_initial = opt_dictionary['theta_initial']()
        rng = np.random.default_rng()
        theta_initial = np.zeros((4))
        theta_initial[0] = np.exp(rng.uniform(-2, 1, 1))
        theta_initial[1] = rng.uniform(0, 1, 1)
        theta_initial[2] = np.exp(rng.uniform(-2, 1, 1))
        theta_initial[3] = rng.uniform(0, 2, 1)
        res = minimize(log_marginal_likelihood_to_optimize(Kernel, X, Y, T, S, targets,noise), x0=theta_initial,
                       method='L-BFGS-B', bounds=opt_dictionary['bounds'],
                       options={'gtol': opt_dictionary['gtol']})
        return res

    
    results = Parallel(n_jobs=n_threads)(delayed(single_optimization_run)(Kernel, X, Y, T, S, targets,noise) for _ in tqdm(range(n_restarts)))

    
    best_result = min(results, key=lambda x: x.fun)

    return best_result

