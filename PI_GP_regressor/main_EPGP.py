import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
from main_class import PhysicsInformedGP_regressor
from kernels.kernel_heat_equation import k_ff_jax, k_fu_jax, k_uu_jax, k_uf_jax, gram_Matrix_jax

def main():
    kernel_list = [gram_Matrix_jax, k_uu_jax, k_uf_jax, k_fu_jax, k_ff_jax]
    #then we define the parameters for the kernel
    hyperparameters = ["l_x", "sigma_f","l_t", "alpha"]
    #now we can define our model. The model needs the kernel list and the hyperparameters, aswell as the timedependence
    model_heat_equation = PhysicsInformedGP_regressor(kernel_list,timedependence=True, params=hyperparameters)
    model_heat_equation.set_name_kernel("Wave_equation")
    #now we create the training data and a validation set
    n_training_points = 10
    noise = [1e-10,1e-10]
    model_heat_equation.set_training_data("PI_GP_regressor/data_files/heat_data.csv ",n_training_points, noise)
    model_heat_equation.plot_raw_data()
    n_validation_points = 500  #for calculating the MSE
    model_heat_equation.set_validation_data(n_validation_points)
    X = model_heat_equation.X
    Y = model_heat_equation.Y
    x_train = model_heat_equation.X[:,0]
    t_train = model_heat_equation.X[:,1]
    u_train = model_heat_equation.u_train


    # def k(x1, x2):
    #     x = x1[:, 0][:, jnp.newaxis]
    #     t = x1[:, 1][:, jnp.newaxis]
    #     x_ = x2[:, 0][jnp.newaxis, :]
    #     t_ = x2[:, 1][jnp.newaxis, :]
    #     denom = 1 + (2 * (t + t_))
    #     return jnp.exp(-(x - x_) ** 2 / (2 * denom)) / jnp.sqrt(denom)
    @jit
    def k_2(x1, x2,sigma,l):
        x, t = x1[0], x1[1]
        x_, t_ = x2[0], x2[1]
        denom = 1 + (2 * (t + t_))
        return jnp.sqrt(2*np.pi)*jnp.exp(-0.5/l**2*(x - x_) ** 2 / (2 * denom)) / jnp.sqrt(denom)
    k_2 = vmap(vmap(k_2, (None, 0)), (0, None))
    k = jit(k_2)
    


    
    def posterior_mean(X_star):
        K = k(X, X) + 1e-6 * jnp.eye(len(X))
        K_s = k(X, X_star)
        K_ss = k(X_star, X_star)
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(L.T, np.linalg.solve(L, u_train))
        f_star = jnp.dot(K_s.T, alpha)
        v = jnp.linalg.solve(L, K_s)
        var_f_star = K_ss - jnp.dot(v.T, v)
        return f_star, var_f_star
    
    n_test_points = 100
    x_star, t_star = np.meshgrid(np.linspace(0, 1, n_test_points), np.linspace(0, 1, n_test_points))
    X_star = np.hstack((x_star.reshape(-1, 1), t_star.reshape(-1, 1)))
    u_pred,var_f_star = posterior_mean(X_star)
    u_pred = u_pred.reshape(n_test_points, n_test_points)


    x_star, t_star = model_heat_equation.raw_data[0].reshape(-1,1), model_heat_equation.raw_data[1].reshape(-1,1)
    u_grid = model_heat_equation.raw_data[2]
    f_grid = model_heat_equation.raw_data[3]
    size = (int(np.sqrt(len(x_star))),int(np.sqrt(len(x_star))))
    fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"})
    ax[0].plot_surface(x_star, t_star, u_pred, cmap="viridis")
    ax[1].plot_surface(x_star, t_star, u_grid, cmap="viridis")
    plt.show()
    
    pass



if __name__ == "__main__":
    main()