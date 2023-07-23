import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from functools import partial
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn

def main():
    noise = 0.01
    X_training, Y_training, X_test, noise = get_data_1d(noise)
    rbf_kernel_second_derivative(X_training[1,:].reshape(-1,1), X_test[1,:].reshape(-1,1), 1/2*1**2)
    plt.plot(X_training, Y_training)

    


def get_data_1d(noise):
    
    c = 1.0        
    A = 0.5       
    k = 2 * np.pi  
    omega = 1.0    
    x = np.linspace(0, 2,100)
    t = np.linspace(0, 2 * np.pi, 1)
    
    noise = 0.01
    X_training = np.array([0,0.1,0.4,0.7,1.2,1.5,1.8,2,2.1]).reshape(-1,1)
    X_training = np.sort(X_training, axis=0)
    Y_training = A*np.sin(k*X_training) +  noise * np.random.randn(*X_training.shape)

    X_test = np.linspace(-0.2,2.2,100).reshape(-1,1)
    return X_training, Y_training, X_test, noise

def rbf_kernel(X1, X2, l=1.0, sigma_f_sq=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f_sq * np.exp(-0.5 / l**2 * sqdist)

# in the sklearn kernel the exponent is (-gamma * sqdist) gamma = 1/2*l**2
# so l = np.sqrt(1/2*gamma) and we need to multiply the kernel with sigma_f_sq

# first i try to implement the second derivarive of the kernel by hand

def rbf_kernel_second_derivative(X1, X2, gamma):
    constant = 2 * gamma
    kernel_values = rbf_kernel_sklearn(X1, X2, gamma)
    d2_kernel_fu = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[1]):
        #first part 2 * gamma * (x1_i - x2_i)**2
        factor_xx = constant * (X1[:,i] - X2[:,i])**2 - 1
        d2_kernel_fu = constant * kernel_values * factor_xx
    return d2_kernel_fu

def posterior_distribution(X, targets,x_test, k,l = 1,sigma_f=1, sigma_n = 1, ):
    K = sigma_f*k(X, X, l) + sigma_n * np.eye(len(X))
    K_s = sigma_f*k(X, x_test,l)
    K_ss = sigma_f*k(x_test, x_test,l)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))

    f_star = np.dot(K_s.T, alpha)
    v = np.linalg.solve(L, K_s)
    var_f_star = K_ss - np.dot(v.T, v)
    
    return f_star, var_f_star

if __name__ == '__main__':
    main()