import numpy as np
from scipy.optimize import minimize


def posterior_distribution(X, targets,x_test, k,l = 1,sigma_f=1, sigma_n = 1, ):
    K = k(X, X, l=l, sigma_f_sq = sigma_f) + sigma_n * np.eye(len(X))
    K_s = k(X, x_test,l=l, sigma_f_sq = sigma_f)
    K_ss = k(x_test, x_test,l=l, sigma_f_sq = sigma_f)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))

    f_star = np.dot(K_s.T, alpha)
    v = np.linalg.solve(L, K_s)
    var_f_star = K_ss - np.dot(v.T, v)
    
    return f_star, var_f_star

def marg_log_likelihood(X, targets, k, sigma_n_sq,theta):
    K = k(X, X, L=theta[0], sigma_f_sq=theta[1])+ sigma_n_sq * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))
    
    marg_log_likelihood = -1/2 * np.dot(targets.T,alpha) - np.sum(np.log(np.diagonal(L))) - len(X)/2 * np.log(2*np.pi)
    return marg_log_likelihood

def grad_marg_log_likelihood(X, targets, k):
    def function_to_optimize(theta):
        return marg_log_likelihood(X, targets, k, theta)


