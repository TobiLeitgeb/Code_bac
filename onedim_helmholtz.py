import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from gaussian_processes_util import plot_gp
import GPy
def main():
# Parameters
    L = 10.0  # Length of the domain
    N = 100  # Number of grid points
    n = 5     # Mode parameter
    A = 0.5
# Discretization
    x = np.linspace(0, L, N)

    # Compute the wave number
    k = n * np.pi / L
    print(k, "k")
# Compute the analytical solution
    def analytical_solution(x, k):
        return np.sin(k * x)

    
    plt.plot(x, analytical_solution(x, k))
    plt.xlabel('Position (x)')
    plt.ylabel('Solution (u)')
    plt.title('Analytical Solution of the Helmholtz Equation')
    plt.grid(True)
    plt.show()

    x_train = np.arange(0,L,0.5).reshape(-1,1)
    x_train = np.sort(x_train, axis=0)

    noise = 0.1

    u_train = analytical_solution(x_train, k) + noise*np.random.randn(len(x_train)).reshape(-1,1)
    plt.scatter(x_train, u_train)

    
    
    
    x_test = np.linspace(-0.2,L,100).reshape(-1,1)

    sigma_n_sq = 0.00001
    
    res = optimization_restarts(10, x_train, u_train, helmholtz_equation_kernel, sigma_n_sq)
    print("gamma = ", res[0], "nu = ", res[1], "sigma_f = ", res[2])
    print(marg_log_likelihood(x_train,u_train,helmholtz_equation_kernel,sigma_n_sq,theta=res))
    mean, cov = posterior_distribution(x_train, u_train, x_test, helmholtz_equation_kernel, gamma = res[0], nu = res[1], sigma_f = res[2], sigma_n = sigma_n_sq)

    plot_gp(mean, cov, x_test, X_train=x_train, Y_train=u_train, samples=[])
    
    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    m = GPy.models.GPRegression(x_train, u_train, kernel)
    #fic the noise variance to known value
    m.Gaussian_noise.variance = sigma_n_sq
    m.Gaussian_noise.variance.fix()
    m.optimize()
    display(m)
    m.plot()

    #mean_2, cov_2 = posterior_distribution(x_train, u_train, x_test, test_helmholtz, l = m.rbf.lengthscale[0], sigma_f = m.rbf.variance[0], sigma_n = sigma_n_sq)
######################functions######################
def helmholtz_equation_kernel(x, y, gamma,nu,sigma_f_sq=1.0):
    kernel_value = rbf_kernel_sklearn(x, y, gamma)
    
    n, m = x.shape[0], y.shape[0]
    dk_ff = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            diff_t = x[i] - y[j]
            double_dev = 16*gamma**4*diff_t**4 - 48*gamma**2*diff_t**2 + 12*gamma**2
            single_dev = 2*nu**2*(4*gamma**2 * diff_t**2 - 2*gamma)
            no_dev = -nu**4
            dk_ff_t = double_dev + single_dev + no_dev
            
            dk_ff[i, j] = dk_ff_t
    return  dk_ff*kernel_value*sigma_f_sq



def test_helmholtz(x,x_prime,alpha,sigma_f_sq=1.0):
    return sigma_f_sq*np.cos(alpha*(x - x_prime))

def ensure_psd(K):
    jitter = 1e-6  # Small constant
    i = 0
    while np.any(np.linalg.eigvals(K) <= 0):
        
        i += 1
        K += np.eye(K.shape[0]) * jitter
        jitter *= 10
    if i > 2:
        print(f"Attention! Added {i} times jitter to K matrix")
    return K

def posterior_distribution(X, targets,x_test, k,gamma = 1,nu = 1, sigma_f=1, sigma_n = 1):
    K = k(X, X, gamma,nu, sigma_f_sq = sigma_f) + sigma_n * np.eye(len(X))
    K_s = k(X, x_test,gamma,nu, sigma_f_sq = sigma_f)
    K_ss = k(x_test, x_test,gamma,nu, sigma_f_sq = sigma_f) 
    ensure_psd(K)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))

    f_star = np.dot(K_s.T, alpha)
    v = np.linalg.solve(L, K_s)
    var_f_star = K_ss - np.dot(v.T, v)
    
    return f_star, var_f_star

def posterior_distribution_test(X, targets,x_test, k,gamma = 1, sigma_f=1, sigma_n = 1):
    K = k(X, X,  sigma_f_sq = sigma_f) + sigma_n * np.eye(len(X))
    K_s = k(X, x_test,gamma, sigma_f_sq = sigma_f)
    K_ss = k(x_test, x_test,gamma, sigma_f_sq = sigma_f)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))

    f_star = np.dot(K_s.T, alpha)
    v = np.linalg.solve(L, K_s)
    var_f_star = K_ss - np.dot(v.T, v)
    
    return f_star, var_f_star

def marg_log_likelihood(X, targets, k, sigma_n_sq,theta):

    K = k(X,X, gamma=theta[0], nu=theta[1], sigma_f_sq = theta[2]) + sigma_n_sq * np.eye(len(X))
    ensure_psd(K)
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


def rbf_kernel(X1, X2, l=1.0, sigma_f_sq=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f_sq * np.exp(-0.5 / l**2 * sqdist)

def rbf_kernel_second_derivative(X1, X2, gamma):
    constant = 2 * gamma
    kernel_values = rbf_kernel_sklearn(X1, X2, gamma)
    d2_kernel_fu = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[1]):
        #first part 2 * gamma * (x1_i - x2_i)**2
        factor_xx = constant * (X1[:,i] - X2[:,i])**2 - 1
        d2_kernel_fu = constant * kernel_values * factor_xx
    return d2_kernel_fu


def create_derivative_matrix(x_train, x_test, gamma):
    derivative_matrix = np.zeros((len(x_train), len(x_test)))
    for i in range(len(x_train)):
        for j in range(len(x_test)):
            derivative_matrix[i,j] = rbf_kernel_second_derivative(x_train[i,:].reshape(-1,1), x_test[j,:].reshape(-1,1), gamma)
    return derivative_matrix

if __name__ == '__main__':
    main()
#