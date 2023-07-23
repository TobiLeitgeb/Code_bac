
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from gaussian_processes_util import plot_gp
import GPy

def main(): 

    m = 3      # Mass
    b = 0.1       # Damping coefficient
    k = 1.0       # Spring constant
    y0 = 1.0      # Initial displacement
    v0 = 0.0      # Initial velocity
    t = np.linspace(0, 10, 1000)

    y_analytical = damped_harmonic_oscillator_analytical(m, b, k, y0, v0, t)

    #get 10 random points
    noise = 0.00001
    number_of_samples = 10
    rng = np.random.default_rng(seed = 5)
    random_numbers = rng.integers(0,int(len(y_analytical)),number_of_samples)
    
    random_numbers = np.sort(random_numbers, axis=0)
    t_training_data = t[random_numbers]
    y_training_data = y_analytical[random_numbers] + noise*np.random.randn(len(random_numbers))

    t_training_data = t_training_data.reshape(-1,1)
    y_training_data = y_training_data.reshape(-1,1)
    t_test_data = t.reshape(-1,1)
    plt.scatter(t_training_data, y_training_data)
    plt.ylabel("Auslenkung")
    plt.xlabel("t")

    res = optimization_restarts(10, t_training_data, y_training_data, damped_harmonic_oscillator_kernel, sigma_n_sq = noise)
    print("gamma = ", res[0], "m = ", res[1], "sigma_f = ", res[2])
    
    #print("gamma = ", res[0], "m = ", res[1], "b = ", res[2], "k_spring = ", res[3], "sigma_f = ", res[4])
    mean, cov = posterior_distribution(t_training_data, y_training_data, t_test_data, damped_harmonic_oscillator_kernel, noise,res)
    plot_gp(mean, cov, t_test_data, X_train=t_training_data, Y_train=y_training_data, samples=[])

    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    m = GPy.models.GPRegression(t_training_data, y_training_data, kernel)
    #fic the noise variance to known value
    m.Gaussian_noise.variance = noise
    m.Gaussian_noise.variance.fix()
    m.optimize()
    display(m)
    m.plot()


    pass
#functions---------------------------------------------------------------------

def damped_harmonic_oscillator_analytical(m, b, k, y0, v0, t):
    omega = np.sqrt(k / m - (b / (2 * m))**2)
    A = np.sqrt(y0**2 + ((v0 + b * y0 / (2 * m)) / omega)**2)
    phi = np.arctan2((v0 + b * y0 / (2 * m)) / omega, y0)
    y = A * np.exp(-b * t / (2 * m)) * np.cos(omega * t + phi)
    return y

def damped_harmonic_oscillator_kernel(X1, X2, theta):
    gamma = 1
    m = theta[1]
    sigma_f_sq = 1
    b = 0.1
    k_spring = 1
    kernel_value = rbf_kernel_sklearn(X1, X2, gamma)
    euc_dist_sq = euclidean_distances(X1, X2)**2
    # second derivative of the kernel
    first_part = 16*gamma**4  * euc_dist_sq**2
    second_part = -48*gamma**3  * euc_dist_sq + 12*gamma**2 * kernel_value
    third_part = 12*gamma**2 
    second_derivative =  (first_part + second_part + third_part + 1) 
    # first derivative of the kernel
    first_derivative = (2*gamma - 4*gamma**2 * euc_dist_sq) * b/m
    no_derivative =  k_spring
    return sigma_f_sq * (first_derivative + second_derivative) * kernel_value * 1/k_spring

def posterior_distribution(X, targets,x_test, k,sigma_n, theta):
    K = k(X, X, theta) + sigma_n * np.eye(len(X))
    K_s = k(X, x_test,theta)
    K_ss = k(x_test, x_test,theta)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))

    f_star = np.dot(K_s.T, alpha)
    v = np.linalg.solve(L, K_s)
    var_f_star = K_ss - np.dot(v.T, v)
    
    return f_star, var_f_star


def marg_log_likelihood(X, targets, k, sigma_n_sq,theta):

    K = k(X,X, theta) + sigma_n_sq * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))
    
    marg_log_likelihood = 1/2 * np.dot(targets.T,alpha) + np.sum(np.log(np.diagonal(L))) + len(X)/2 * np.log(2*np.pi)
    return marg_log_likelihood

def grad_marg_log_likelihood(X, targets, k, sigma_n_sq):
    def function_to_optimize(theta):
        mll = marg_log_likelihood(X, targets, k,sigma_n_sq, theta)
        return mll
    return function_to_optimize

def optimization_restarts(n_restarts, X, targets, k, sigma_n_sq):
    best_mll = np.inf
    best_theta = np.zeros((3))
    for i in range(n_restarts):
        theta_initial = np.random.rand(3)
        res = minimize(grad_marg_log_likelihood(X,targets,k,sigma_n_sq), x0=theta_initial, 
                       method='L-BFGS-B', bounds=((1e-5, None), (1e-9, None), (1e-7, None)))
        if res.fun < best_mll:
            best_mll = res.fun
            best_theta = res.x
    return best_theta




if __name__ == "__main__":
    main()