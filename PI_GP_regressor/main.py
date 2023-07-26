from main_class import PhysicsInformedGP_regressor

from kernel_helmholtz import gram_Matrix, k_ff, k_uf, k_fu, k_uu
import numpy as np

def main():
    #set up the GP
    Kernel = [gram_Matrix, k_uu, k_uf, k_fu, k_ff]
    
    params = ["l_x", "sigma_f_sq", "l_t", "c"]
    noise = [0.00001,0.00001]
    n_training_points = 25
    model = PhysicsInformedGP_regressor(kernel =Kernel, timedependence = True,params=params)
    model.set_training_data("gaussian_f_c3.csv", n_training_points, noise)
    
    dictionary_BFGS_CG_TNC = {'theta_initial': get_initial_values,   #needed for all optimization methods   
                            'bounds': ((1e-2, None), (1e-2, None), (1e-1, None),(1e-2, None)), #needed for TNC and L-BFGS-B
                            'gtol': 1e-6}
    
    restarts, threads = 100, -1
    model.train("CG",restarts,threads,dictionary_BFGS_CG_TNC)
    x_star, t_star = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))

    X_star = np.hstack((x_star.reshape(-1,1),t_star.reshape(-1,1)))
    model.predict_u(X_star)
    model.predict_f(X_star)
    model.plot_prediction(X_star,"test","prediction_2d.png")
    model.plot_variance(X_star,"test","variance_2d.png")
    model.set_validation_data(n_validation_points = 1000)
    model.error()
    model.plot_difference("test","difference_2d.png")
    print(model)
    pass


def get_initial_values():
    """returns the initial values for the hyperparameters
    for the length scales we initialize them randomly as log(l) ~ U(-2.5,1)
    """
    rng = np.random.default_rng()
    theta_initial = np.zeros((4))
    theta_initial[0] = np.exp(rng.uniform(-1.3, 0.3, 1))
    theta_initial[1] = rng.uniform(0, 2, 1)
    theta_initial[2] = np.exp(rng.uniform(-1.3, 0.3, 1))
    theta_initial[3] = rng.uniform(1, 4, 1)
    return theta_initial



if __name__ == "__main__":
    main()