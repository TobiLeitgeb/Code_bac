from kernels.kernel_oscillator import k_ff, k_uf, k_fu, k_uf, gram_Matrix, k_uu
#from kernels.kernel_oscillator import k_ff, k_uf, k_fu, k_uf, gram_Matrix, k_uu

from main_class import PhysicsInformedGP_regressor
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def main():
    kernel = [gram_Matrix,k_uu,k_uf,k_fu,k_ff]
    params = ["l","sigma_f","b","k"]
    seeds_training = [50,14] #sets the seeds for the random training points--- change when the points are not optimal
    model = PhysicsInformedGP_regressor(kernel,timedependence=False, params = params)
    model.set_training_data("PI_GP_regressor/data_files/damped_m1k2b1.csv",8,[1e-7,1e-7],seeds_training)
    model.plot_raw_data(Training_points=True)
    model.set_validation_data(1000)
    model.plot_raw_data(Training_points=False)
    plt.show()
    def get_initial_values():
        """returns the initial values for the hyperparameters
        for the length scales we initialize them randomly as log(l) ~ U(-2.5,1)
        """
        rng = np.random.default_rng()
        theta_initial = np.zeros((4))
        theta_initial[0] = np.exp(rng.uniform(-1.3, 0.3, 1))
        theta_initial[1] = rng.uniform(0, 2, 1)
        theta_initial[2] = rng.uniform(1, 3, 1)
        theta_initial[3] = rng.uniform(1, 3, 1)
        #theta_initial[4] = rng.uniform(0, 4, 1)
        return theta_initial
    n_iterations, n_threads = 40, 10

    model.train("TNC",n_iterations,n_threads,{'theta_initial': get_initial_values,   #needed for all optimization methods
                                              'bounds': ((1e-2, None), (1e-5, None), (1e-2, None),(1e-2, None)),#,(1e-2,None)), #needed for TNC and L-BFGS-B
                                              'gtol': 1e-6})
      
    X_star = np.linspace(0,3,100).reshape(-1,1)
    model.predict_model(X_star)
    model.plot_prediction(X_star, "prediction", "PI_GP_regressor/plots/oscillator/prediction.png")
    model.error()
    print(model)
    model.use_GPy(X_star, "PI_GP_regressor/plots/oscillator/GPy_model.png")
    
    plot_kernel_mat(model.k_ff(X_star,X_star,model.get_params()))
    plot_kernel_mat(model.k_uu(X_star,X_star,model.get_params()))
    plot_kernel_mat(model.k_uf(X_star,X_star,model.get_params()))
    plot_kernel_mat(model.k_fu(X_star,X_star,model.get_params()))
    pass

def plot_kernel_mat(K):
    # plot
    plt.figure()
    cont = plt.imshow(K, cmap='Reds')
    plt.colorbar(cont)
    plt.title(r'$K_{ff}$, (rbf)', fontsize=20, weight='bold')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

    
