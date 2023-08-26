from main_class import PhysicsInformedGP_regressor
from kernels.kernel_wave_eq import gram_Matrix_jax,k_ff_jax, k_fu_jax, k_uf_jax, k_uu,gram_Matrix, k_ff, k_fu, k_uf
import numpy as np
import matplotlib.pyplot as plt
def main():
    # first we put the kernel parts into a list
    kernel_list_jax = [gram_Matrix_jax, k_uu, k_uf_jax, k_fu_jax, k_ff_jax]
    kernel_list = [gram_Matrix, k_uu, k_uf, k_fu, k_ff]
    #then we define the parameters for the kernel
    hyperparameters = ["l_x", "sigma_f","l_t", "c"]
    #now we can define our model. The model needs the kernel list and the hyperparameters, aswell as the timedependence
    model_wave_equation = PhysicsInformedGP_regressor(kernel_list,timedependence=True, params=hyperparameters)
    model_wave_equation.set_name_kernel("Wave_equation")
    #now we create the training data and a validation set
    n_training_points = 20
    noise = [1e-8,1e-8]
    model_wave_equation.set_training_data("PI_GP_regressor/data_files/wave_second.csv ",n_training_points, noise)

    n_validation_points = 500  #for calculating the MSE
    model_wave_equation.set_validation_data(n_validation_points)
    model_wave_equation.plot_raw_data()
    # plt.show()
    #for the training we first need to define the initial parameters for the restarts
    def get_initial_values():
        """returns the initial values for the hyperparameters
        for the length scales we initialize them randomly as log(l) ~ U(-1.3,1)
        """
        rng = np.random.default_rng()
        theta_initial = np.zeros((4))
        theta_initial[0] = np.exp(rng.uniform(-1.3, 0.3, 1))  #lx
        theta_initial[1] = rng.uniform(0, 1, 1)               #sigma_f
        theta_initial[2] = np.exp(rng.uniform(-1.3, 0.3, 1))  #lt
        theta_initial[3] = rng.uniform(1, 4, 1)               #c
        return theta_initial
    
    #now we can train the model. We can choose different methods for the training. 
    # CG: conjugate gradient --- fast but often not very accurate
    # TNC: truncated Newton --- slower but more accurate
    # L-BFGS-B: limited memory BFGS --- fast and accurate (not always for some reason)
    #generally TNC is the best choice
    n_restarts = 10
    n_threads = 8
    opt_params_dict = {'theta_initial': get_initial_values,   #needed for all optimization methods
                       'bounds': ((1e-2, None), (1e-5, None), (1e-3, None),(1e-2, None)), #needed for TNC and L-BFGS-B
                       'gtol': 1e-6}
    
    model_wave_equation.train("CG",n_restarts, n_threads,opt_params_dict)
    #an alternative way is to set the parameters manually
    #model_wave_equation.set_parameters([0.1,0.1,0.1,0.1])
    
    #now we can make predictions with the model
    n_test_points = 100
    x_star, t_star = np.meshgrid(np.linspace(0, 1, n_test_points), np.linspace(0, 1, n_test_points))
    X_star = np.hstack((x_star.reshape(-1, 1), t_star.reshape(-1, 1)))
    model_wave_equation.predict_model(X_star)
    #now we can plot the results
    #predictive mean
    heat_map = True
    name_pred, save_path = "Predictive mean $\\overline{ f_*}$","PI_GP_regressor/plots/wave_eq/predictive_mean.png"
    model_wave_equation.plot_prediction(X_star, name_pred, save_path,heat_map=heat_map)
    #predictive variance
    name_var, save_path = "Predictive variance $\\sigma_*^2$","PI_GP_regressor/plots/wave_eq/predictive_variance.png"
    model_wave_equation.plot_variance(X_star, name_var, save_path)
    #difference to the numerical solution(anlytical solution)#
    name_diff, save_path = "Difference to the analytical solution ","PI_GP_regressor/plots/wave_eq/difference.png"
    model_wave_equation.plot_difference(name_diff, save_path)

    #now we calculate the MSE for the validation set
    model_wave_equation.error()
    print(model_wave_equation)
    #now we can predict our model using a normal rfb kernel with the GPy library
    save_path_gpy = "PI_GP_regressor/plots/wave_eq/predictive_mean_GPy.png"
    model_wave_equation.use_GPy(X_star,save_path_gpy,heat_map=heat_map)
    print("-------------GPy-------------")
    model_wave_equation.plot_difference_GPy("difference GPy ", "PI_GP_regressor/plots/wave_eq/difference_GPy.png")
    model_wave_equation.plot_variance_GPy("predictive variance GPy ", "PI_GP_regressor/plots/wave_eq/predictive_variance_GPy.png")
    
    pass



if __name__ == "__main__":
    main()