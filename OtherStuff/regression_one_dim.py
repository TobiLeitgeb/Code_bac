import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import GPy
from regresssion import posterior_distribution, marg_log_likelihood, grad_marg_log_likelihood
from gaussian_processes_util import plot_data_2D, plot_gp_2D, plot_data_1D, plot_gp
from sklearn.metrics.pairwise import rbf_kernel
#create onedimensional data for a diffusion equation

def get_data_1d():
    # Parameters
    c = 1.0        # Wave velocity
    A = 0.5        # Amplitude of forcing term
    k = 2 * np.pi  # Wave number
    omega = 1.0    # Angular frequency
    x = np.linspace(0, 2, 10)
    t = np.linspace(0, 2 * np.pi, 1)
    #gx, gy = np.meshgrid(x, t)
    #X_2D = np.c_[gx.ravel(), gy.ravel()]
    #Calculate the displacement u(x, t) at each point
    # u = np.zeros((len(t), len(x)))
    # for i, ti in enumerate(t):
    #     for j, xi in enumerate(x):
    #         u[i, j] = A * np.sin(k * xi - omega * ti)
    # Plot surface
    #plot_gp_2D(gx, gy, u, X_2D, u, 'Initial data', 1)
    #plt.show()
    noise = 0.01
    X_training = np.array([0,0.1,0.4,0.7,1.2,1.5,1.8,2,2.1]).reshape(-1,1)
    X_training = np.sort(X_training, axis=0)
    Y_training = A*np.sin(k*X_training) +  noise * np.random.randn(*X_training.shape)

    X_test = np.linspace(-0.2,2.2,100).reshape(-1,1)
    return X_training, Y_training, X_test, noise
X_training, Y_training, X_test, noise = get_data_1d()

plt.plot(X_training, Y_training, 'rx')
# comute mean and variance
f_star, var_f_star = posterior_distribution(X_training, Y_training, X_test, rbf_kernel)
#plt.plot(X_test, f_star, label='Mean')

sigma_n_sq = noise**2


res = minimize(grad_marg_log_likelihood(X_training, Y_training, rbf_kernel,sigma_n_sq), [1,1], 
               method='L-BFGS-B', bounds=((1e-5, None), (1e-5, None)))

theta_opt = res.x
print(f"Optimized theta: l = {theta_opt[0]}, sigma_f_sq = {theta_opt[1]}")

f_star, var_f_star = posterior_distribution(X_training, Y_training, X_test, rbf_kernel, 
                                            l = theta_opt[0], sigma_f = theta_opt[1], sigma_n = sigma_n_sq)

samples = np.random.multivariate_normal(f_star.ravel(), var_f_star, 3)
plt.plot(np.linspace(-0.1,2.2,100), 0.5*np.sin(2*np.pi*np.linspace(-0.1,2.2,100)),label = 'true function')
plot_gp(f_star, var_f_star, X_test,samples=[])

# kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
# m = GPy.models.GPRegression(X_training, Y_training, kernel)
# m.Gaussian_noise.variance = 0.01
# m.Gaussian_noise.variance.fix()
# m.optimize()
# display(m)
# m.plot()