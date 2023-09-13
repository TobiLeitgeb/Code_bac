from gram_matrix import k_uu, k_uf, k_ff,k_fu
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from jax import jit, grad, vmap
import jax.numpy as jnp

def posterior_distribution_u(X_u,X_f,T_u, T_f, x_star, t_star, targets, noise, params, Kernel:callable):
    """returns the mean and covariance matrix of the posterior distribution"""
    K = Kernel(X_u, X_f, T_u, T_f, params, noise)
    L = jnp.linalg.cholesky(K + 1e-6 * jnp.zeros(K.shape)) #add some jitter for stability
    q_1 = k_uu(x_star, X_u, t_star, T_u, params) 
    q_2 = k_uf(x_star, X_f, t_star, T_f, params)
    q = jnp.hstack((q_1,q_2))
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, targets))
    f_star = jnp.dot(q,alpha)

    alpha_var = jnp.linalg.solve(L.T, jnp.linalg.solve(L, q.T))
    cov_f_star = k_uu(x_star,x_star,t_star,t_star, params) - q@alpha_var
    var = jnp.diag(cov_f_star)
    return f_star, var

def posterior_distribution_f(X_u,X_f,T_u, T_f, x_star, t_star, targets, noise, params, Kernel:callable):
    """returns the mean and covariance matrix of the posterior distribution"""
    K = Kernel(X_u, X_f, T_u, T_f, params, noise)
    L = jnp.linalg.cholesky(K + 1e-6 * jnp.zeros(K.shape)) #add some jitter for stability
    q_1 = k_fu(x_star, X_u, t_star, T_u, params) 
    q_2 = k_ff(x_star, X_f, t_star, T_f, params)
    q = jnp.hstack((q_1,q_2))
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, targets))
    f_star = jnp.dot(q,alpha)
    
    alpha_var = jnp.linalg.solve(L.T, jnp.linalg.solve(L, q.T))
    cov_f_star = k_ff(x_star,x_star,t_star,t_star, params) - q@alpha_var
    var = jnp.diag(cov_f_star)
    return f_star, var