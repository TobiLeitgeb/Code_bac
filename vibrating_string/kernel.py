import jax.numpy as jnp
from jax import jit, vmap, grad

@jit
def rbf_kernel_single_x(x: float, y: float, params: list) -> float:
    """general RBF kernel k(x,y)"""
    l_x, sigma_f_sq = params[0], params[1]
    sqdist = jnp.sum(x-y)**2
    return sigma_f_sq * jnp.exp(-0.5 / l_x**2 * sqdist)
@jit
def rbf_kernel_single_t(t: float, s: float, l_t: float) -> float:
    """general RBF kernel. takes scalar inputs t,s and returns k(t,s)"""
    sqdist = jnp.sum(t-s)**2
    value = jnp.exp(-0.5 / l_t**2 * sqdist)
    return value

@jit
def k_uu(X, Y, T, S, params):
    """
    computes k_uu part of the block matrix K
    params = [l_x, sigma_f_sq, l_t, c]
    """
    # vectorize the kernel so that it can take arrays as input
    vectorized_rbf_kernel_x = vmap(vmap(rbf_kernel_single_x, (None, 0, None)), (0, None, None)) 
    vectorized_rbf_kernel_t = vmap(vmap(rbf_kernel_single_t, (None, 0, None)), (0, None, None)) 
    params = params[:-1] #remove c from the parameters since it is not used in the kernel
    l_t = params[2]
    #combine the two kernels to the full kernel k(x,t,y,s)
    return vectorized_rbf_kernel_x(X, Y, params) * vectorized_rbf_kernel_t(T, S, l_t)

#describtion in the notebook new_kernel_notebook

@jit
def k_x_welle(x:float, y:float, params):
    """computes the k_x part of the derivative k_ttyy. The parts are seperated to make vmap work"""
    gamma_x = 0.5 / params[0]**2
    polynom = (2*gamma_x*(x-y)**2 - 1) 
    return polynom * rbf_kernel_single_x(x,y,params) * 2*gamma_x
@jit
def k_t_welle(t:float, s:float, params):
    """computes the k_t part of the derivative k_ttyy. The parts are seperated to make vmap work"""
    gamma_t = 0.5 / params[2]**2
    polynom = (2*gamma_t*(t-s)**2 - 1) 
    return polynom * rbf_kernel_single_t(t,s,params[2]) * 2*gamma_t
@jit
def k_ff(X, Y, T, S, params):
    """computes k_ff part of the block matrix K. It corresponds to the part with double the operator L: k_ff = L k_uu L'^T
    #k_ff =  d^2/dx^2 d^2/dy^2 K_uu - 2/c^2 d^2/dt^2 d^2/dy^2 K_uu + 1/c^4 d^2/dt^2 d^2/ds^2 K_uu
       params = [l_x, sigma_f_sq, l_t, c]
    """
    c = params[-1]
    l_t = params[2]
    params = params[:-1]
    # flatten the data so that it can be used in the grad function (only 1d arrays are allowed in grad)
    X,Y,T,S = X.flatten(), Y.flatten(), T.flatten(), S.flatten() 
    #vectorizazion of both the kernel functions 
    rbf_kernel_x = vmap(vmap(rbf_kernel_single_x, (None, 0, None)), (0, None, None))
    rbf_kernel_t = vmap(vmap(rbf_kernel_single_t, (None, 0, None)), (0, None, None))
    #compute the derivatives seperately and then multiply by the other kernel.
    # d^2/dx^2 d^2/dy^2 K_uu
    dk_dydy = grad(grad(rbf_kernel_single_x, argnums = 1), argnums = 1) #second derivative of  k with respect y
    dk_dxdxdydy = grad(grad(dk_dydy, argnums = 0), argnums = 0) # second derivative with respect to x of dk_dydy
    vectorized_dxdxdydy = vmap(vmap(dk_dxdxdydy, (None, 0, None)), (0, None, None))
    k_xxyy = vectorized_dxdxdydy(X,Y,params) * rbf_kernel_t(T,S,l_t)

    #d^2/dt^2 d^2/dy^2 K_uu
    vec_k_x_welle = vmap(vmap(k_x_welle, (None, 0, None)), (0, None, None))(X,Y,params)
    vec_k_t_welle = vmap(vmap(k_t_welle, (None, 0, None)), (0, None, None))(T,S,params)
    k_ttyy = vec_k_x_welle * vec_k_t_welle

    #d^2/dt^2 d^2/ds^2 K_uu
    dk_ss = grad(grad(rbf_kernel_single_t, argnums = 1), argnums = 1) #second derivative of  k with respect s
    dk_dtdtdsds = grad(grad(dk_ss, argnums = 0), argnums = 0) # second derivative with respect to t of dk_ss
    vectorized_dtdtdsds = vmap(vmap(dk_dtdtdsds, (None, 0, None)), (0, None, None))
    k_ttdsds = vectorized_dtdtdsds(T,S,params[2]) * rbf_kernel_x(X,Y,params)

    return  c**4*k_xxyy - 2*c**2 * k_ttyy +  k_ttdsds


@jit
def k_uf(X,Y,T,S,params):
    c = params[-1]
    l_t = params[2]
    params = params[:-1]
    X,Y,T,S = X.flatten(), Y.flatten(), T.flatten(), S.flatten() 
    rbf_kernel_x = vmap(vmap(rbf_kernel_single_x, (None, 0, None)), (0, None, None))
    rbf_kernel_t = vmap(vmap(rbf_kernel_single_t, (None, 0, None)), (0, None, None))

    #d^2/dy^2 K_uu
    dk_dydy = grad(grad(rbf_kernel_single_x, argnums = 1), argnums = 1) #second derivative of  k with respect y
    vectorized_dydy = vmap(vmap(dk_dydy, (None, 0, None)), (0, None, None))
    k_yy = vectorized_dydy(X,Y,params) * rbf_kernel_t(T,S,l_t)
    #d^2/ds^2 K_uu
    dk_ss = grad(grad(rbf_kernel_single_t, argnums = 1), argnums = 1) #second derivative of  k with respect s
    vectorized_dss = vmap(vmap(dk_ss, (None, 0, None)), (0, None, None))
    k_ss = vectorized_dss(T,S,params[2]) * rbf_kernel_x(X,Y,params)

    return -c**2*k_yy + 1 *  k_ss

@jit
def k_fu(X,Y,T,S,params):
    c = params[-1]
    l_t = params[2]
    params = params[:-1]
    X,Y,T,S = X.flatten(), Y.flatten(), T.flatten(), S.flatten() 
    rbf_kernel_x = vmap(vmap(rbf_kernel_single_x, (None, 0, None)), (0, None, None))
    rbf_kernel_t = vmap(vmap(rbf_kernel_single_t, (None, 0, None)), (0, None, None))

    #d^2/dx^2 K_uu
    dk_dxdx = grad(grad(rbf_kernel_single_x, argnums = 0), argnums = 0) #second derivative of  k with respect y
    vectorized_dxdx = vmap(vmap(dk_dxdx, (None, 0, None)), (0, None, None))
    k_xx = vectorized_dxdx(X,Y,params) * rbf_kernel_t(T,S,l_t)
    #d^2/dt^2 K_uu
    dk_tt = grad(grad(rbf_kernel_single_t, argnums = 0), argnums = 0) #second derivative of  k with respect s
    vectorized_dtt = vmap(vmap(dk_tt, (None, 0, None)), (0, None, None))
    k_tt = vectorized_dtt(T,S,params[2]) * rbf_kernel_x(X,Y,params)

    return -c**2*k_xx + 1 * k_tt
 
@jit
def gram_Matrix(X, Y, T, S, params, noise = [0,0]):
    """computes the gram matrix of the kernel
    params = [l_x, sigma_f_sq, l_t, alpha]
    noise = [noise_u, noise_f]
    """
    k_uu_matrix = k_uu(X, X, T, T, params) + noise[0] * jnp.eye(len(X))
    k_uf_matrix = k_uf(X, Y, T, S, params)
    k_fu_matrix = k_fu(Y, X, S, T, params) 
    k_ff_matrix = k_ff(Y, Y, S, S, params) + noise[1] * jnp.eye(len(Y))
    #combine all the matrices to the full gram matrix
    K = jnp.block([[k_uu_matrix, k_uf_matrix], [k_fu_matrix, k_ff_matrix]])
    return K