import jax.numpy as jnp
from jax import grad, jit, vmap

"""
params = [sigma, l_space, l_t] We use the same length scale for space and a seperate one for time
"""


def single_rbf(x, x_bar, params):
    """Single RBF kernel function for a 3 dimensional input.
    """
    x,y,t = x[0], x[1], x[2]
    x_bar, y_bar, t_bar = x_bar[0], x_bar[1], x_bar[2]
    return params[1]*jnp.exp( -(((x-x_bar)**2+ (y-y_bar)**2))/ (2 * params[0]**2) + (t-t_bar)**2 / (2*params[2]**2))

k_uu = jit(vmap(vmap(single_rbf,(None,0,None)), (0,None,None)))


# test the single rbf function
X_test = jnp.array([[1,2,1],
                    [3,4,5]])
X_bar_test = jnp.array([[1,2,1],
                        [3,4,1]])
params_test = [1,1,1]
print(k_uu(X_test, X_bar_test, params_test))

def k_uf(x, x_bar, params):
    """ Kernel function for the mixed covaricen function L_x' k_uu = k_uf U x F --> R
        the derivatives were calcualted with exp(-gamma ...) --> gamma = 1/(2*sigma^2)"""
    k_uu_data = single_rbf(x, x_bar, params)
    x,y,t = x[0], x[1], x[2]
    x_bar, y_bar, t_bar = x_bar[0], x_bar[1], x_bar[2]
    gamma = 1/(2*params[0]**2)
    prefactor = 2*gamma *(2*gamma*((x-x_bar)**2 + (y-y_bar)**2)-2)
    return prefactor* k_uu_data