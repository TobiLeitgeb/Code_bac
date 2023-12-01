# Code for my bachelor thesis.
main_class contains the class for the physics informed model. It has functionality for reading data, setting the training data, setting and optimizing the hyperparameters, calculating the predictions and plotting the results.
The kernel folder contains specific kernels for different ODEs. Before using the class we need a .csv file containing the ground truth. From this the training data gets sampled.
# A general way of using the class is presented here:


```python
kernel_list = [gram_Matrix, k_uu, k_uf, k_fu, k_ff]
hyperparameters = ["l_x", "sigma_f"]  # list of the hyperparameters
model = PhysicsInformedGP_regressor(kernel_list,timedependence=True, params = hyperparameters,Dimensions=2)  #initialize the model
model.set_name_kernel("poisson")

n_training_points, noise_sq = 25, [1e-8,1e-8]  
model.set_training_data("poisson_data.csv",n_training_points, noise_sq) #set the training data with a filename 

model.set_validation_data(1000)
model.plot_raw_data()

#here we optimize the hyperparameters
model.jitter = 1e-7
def get_initial_values():
    """returns the initial values for the hyperparameters
    for the length scales we initialize them randomly as log(l) ~ U(-1.3,1)
    """
    rng = np.random.default_rng()
    theta_initial = np.zeros((2))
    theta_initial[0] = np.exp(rng.uniform(-1.3, 0.4, 1))  #lx
    theta_initial[1] = rng.uniform(0, 1, 1)               #sigma_f              
    return theta_initial
n_restarts = 100
n_threads = 2
opt_params_dict = {'theta_initial': get_initial_values,   #needed for all optimization methods
                    'bounds': ((1e-2, None), (1e-5, None)), #needed for TNC and L-BFGS-B
                    'gtol': 1e-7}
#model.train("Nelder-Mead",n_restarts, n_threads,opt_params_dict)
model.train("Nelder-Mead",n_restarts, n_threads,opt_params_dict)

and now we can calculate the predictions for a given domain
n_test_points = 100
x_star, t_star = np.meshgrid(np.linspace(0, 1, n_test_points), np.linspace(0, 1, n_test_points))
X_star = np.hstack((x_star.flatten()[:, None], t_star.flatten()[:, None]))
model.predict_model(X_star)