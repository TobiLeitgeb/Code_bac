import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np



@jit
def rbf_kernel_single(x1, x2, params):
    x1, x2 = x1.flatten(), x2.flatten()
    l, sigma_f = params[0], params[1]
    return jnp.squeeze(sigma_f**2 * jnp.exp(-0.5/l**2 * jnp.sum((x1 - x2)**2)))

#mx'' + bx' + kx = f(t)

#k_uu
@jit
def k_uu(X, Y, params):
    X, Y = X.flatten(), Y.flatten()
    vec_rbf_kernel = vmap(vmap(rbf_kernel_single, (None, 0, None)), (0, None, None))
    return vec_rbf_kernel(X, Y, params)

#k_ff
#k_ff = m^2 dt^2 dt'^2 + 2mk dt'^2 + b^2 dtdt' + k^2
@jit
def k_ff(X, Y, params):
    X, Y = X.flatten(), Y.flatten()
    #m = params[2]
    m = 1
    b = params[2]
    k = params[3]
    #dt^2 dt'^2
    dk_yy = grad(grad(rbf_kernel_single, argnums=1), argnums=1)
    dk_xxyy = grad(grad(dk_yy, argnums=0), argnums=(0))
    k_xxyy = vmap(vmap(dk_xxyy, (None, 0, None)), (0, None, None))
    #dt'^2
    dk_yy = grad(grad(rbf_kernel_single, argnums=1), argnums=1)
    k_yy = vmap(vmap(dk_yy, (None, 0, None)), (0, None, None))
    #dtdt'
    dk_xy = grad(grad(rbf_kernel_single, argnums=1), argnums=(0))
    k_xy = vmap(vmap(dk_xy, (None, 0, None)), (0, None, None))
    #k^2
    k_normal = vmap(vmap(rbf_kernel_single, (None, 0, None)), (0, None, None))
    return m**2 * k_xxyy(X, Y, params) + 2*m*k*k_yy(X, Y, params) + b**2 * k_xy(X, Y, params) + k**2 * k_normal(X, Y, params)

#k_uf
#k_uf = m dt'^2 + b dt' + k
@jit
def k_uf(X, Y, params):
    X, Y = X.flatten(), Y.flatten()
    m = 1
    b = params[2]
    k = params[3]
    #dt'^2
    dk_yy = grad(grad(rbf_kernel_single, argnums=1), argnums=1)
    k_yy = vmap(vmap(dk_yy, (None, 0, None)), (0, None, None))
    #dt'
    dk_y = grad(rbf_kernel_single, argnums=1)
    k_y = vmap(vmap(dk_y, (None, 0, None)), (0, None, None))
    #k
    k_normal = vmap(vmap(rbf_kernel_single, (None, 0, None)), (0, None, None))
    return m * k_yy(X, Y, params) + b * k_y(X, Y, params) + k * k_normal(X, Y, params)

@jit
def k_fu(X, Y, params):
    X, Y = X.flatten(), Y.flatten()
    m = 1
    b = params[2]
    k = params[3]
    #dt'^2
    dk_xx = grad(grad(rbf_kernel_single, argnums=0), argnums=0)
    k_xx = vmap(vmap(dk_xx, (None, 0, None)), (0, None, None))
    #dt'
    dk_x = grad(rbf_kernel_single, argnums=0)
    k_x = vmap(vmap(dk_x, (None, 0, None)), (0, None, None))
    #k
    k_normal = vmap(vmap(rbf_kernel_single, (None, 0, None)), (0, None, None))
    return m * k_xx(X, Y, params) + b * k_x(X, Y, params) + k * k_normal(X, Y, params)



@jit
def gram_Matrix(X, Y, params, noise = [0,0]):
    """computes the gram matrix of the kernel
    params = [l_x, sigma_f_sq, l_t, alpha]
    noise = [noise_u, noise_f]
    """
    k_uu_matrix = k_uu(X, X, params) + noise[0] * jnp.eye(len(X))
    k_uf_matrix = k_uf(X, Y, params)
    k_fu_matrix = k_uf_matrix.T
    k_ff_matrix = k_ff(Y, Y, params) + noise[1] * jnp.eye(len(Y))
    #combine all the matrices to the full gram matrix
    K = jnp.block([[k_uu_matrix, k_uf_matrix], [k_fu_matrix, k_ff_matrix]])
    return K