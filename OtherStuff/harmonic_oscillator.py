import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn, euclidean_distances
import GPy

A = 1.0     # amplitude
w = 6    # angular frequency
phi = 0   # phase


t_test = np.linspace(0, 1, 100).reshape(-1,1)  # 100 points evenly spaced over [0,1]


def y(t, A, w, phi):
    return A * np.sin(k*t + phi)
rng_u = np.random.default_rng(seed=42)
t_train = rng_u.uniform(0,1,10)
t_train = np.sort(t_train).reshape(-1,1)
y_train = y(t_train, A, w, phi) 
# Add some noise to simulate measurement errors
noise = np.random.normal(0, 0.1, y_train.shape)
y_train = y_train + noise
y_train = y_train.reshape(-1,1)
# Plot the data
plt.plot(t_train, y_train, 'k.')
plt.plot(t_test, y(t_test,A,w,phi), 'r-')
plt.xlabel('Time (t)')
plt.ylabel('Position (y)')
plt.show()

def dk_ff(t, t_bar, gamma, omega):
        n, m = t.shape[0], t_bar.shape[0]
        dk_ff = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                diff_t = t[i] - t_bar[j]
                double_dev = 16*gamma**4*diff_t**4 - 48*gamma**2*diff_t**2 + 12*gamma**2
                single_dev = 2*omega**2*(4*gamma**2 * diff_t**2 - 2*gamma)
                no_dev = omega**4
                dk_ff_t = double_dev + single_dev + no_dev
                
                dk_ff[i, j] = dk_ff_t
        return  dk_ff


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
def kernel(X,X_bar,gamma, sigma_f):
    return sigma_f**2 * rbf_kernel_sklearn(X,X_bar, gamma)

def K_matrix(t, t_bar, gamma, sigma_f,omega, sigma_n = 0.001):
    k_uu = kernel(t,t_bar, gamma , sigma_f)*dk_ff(t,t_bar, gamma, omega)  + sigma_n * np.eye(len(t))
    ensure_psd(k_uu)
    return k_uu

def marginal_likelihood(K_matrix,t_train,targets,noise, theta):
    gamma, sigma_f, omega = theta[0], theta[1], theta[2]
    K = K_matrix(t_train, t_train, gamma, sigma_f, omega, noise[0])
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))
    return 0.5 * y_train.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y_train) * np.log(2*np.pi)

def grad_marg_log_likelihood(K_matrix: callable, X, targets, noise: list):
    def function_to_optimize(theta):
        mll = marginal_likelihood(K_matrix, X, targets, noise, theta)
        return mll
    return function_to_optimize

res = minimize(grad_marg_log_likelihood(K_matrix, t_train, y_train, [0.001]), x0=[0.1,0.1,1], 
               method='L-BFGS-B', bounds=((1e-5, None), (1e-5, None), (1e-5, None)))
print(res.x)
targets = y_train
t_test = np.linspace(0, 1, 100).reshape(-1,1) 

gamma = res.x[0]
sigma_f = res.x[1]
omega = res.x[2]

k = kernel(t_train, t_train, gamma,sigma_f) * dk_ff(t_train, t_train, gamma, omega) + 0.001 * np.eye(len(t_train))
k_star = kernel(t_train, t_test, gamma,sigma_f) * dk_ff(t_train, t_test, gamma, omega)
k_star_star = kernel(t_test, t_test, gamma,sigma_f) * dk_ff(t_test, t_test, gamma, omega)

ensure_psd(k)
L = np.linalg.cholesky(k)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))
print(k_star_star.shape)
f_star = np.dot(k_star.T, alpha)
plt.plot(t_test, f_star, 'r-')
plt.plot(t_train, y_train, 'k.')

# kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
# m = GPy.models.GPRegression(t_train, y_train, kernel)
# #fic the noise variance to known value
# m.Gaussian_noise.variance = 0.001
# m.Gaussian_noise.variance.fix()
# m.optimize()
# display(m)
# m.plot()
