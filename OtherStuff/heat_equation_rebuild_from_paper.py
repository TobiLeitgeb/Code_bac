import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn, euclidean_distances
import sympy as sp

def main():
    # recreate the data from the paper
    alpha_real = 1
    noise = [0.0001, 0.0001]
    u_values, f_values, t_u, x_u, t_f, x_f = get_data_set(20,noise)

    # plot the general solution and the data
    plot_data_3d(u_values, f_values, t_u, x_u, t_f, x_f)
    

    # define the kernel function
    
   
    
    
    

    targets = np.vstack((u_values, f_values))
    res = optimization_restarts(create_K_matrix, 1, x_u, x_f, t_u, t_f, targets, noise)
    print(res)
    
    thetha = res
    K_optimized = create_K_matrix(x_u, x_f, t_u, t_f, noise[0], noise[1], thetha[0], thetha[1], thetha[2], thetha[3])
    L = np.linalg.cholesky(K_optimized)


    X_test = np.linspace(0,1,100).reshape(-1,1)
    t_test = np.linspace(0,1,100).reshape(-1,1)
    x_star, t_star = np.meshgrid(X_test, t_test)
    x_star = x_star.reshape(-1,1)
    t_star = t_star.reshape(-1,1)


    q_1 = kernel_rbf(x_star,x_u,t_star,t_u,thetha[0], thetha[1], thetha[2]) 
    q_2 = kernel_rbf(x_star,x_f,t_star,t_f, thetha[0], thetha[1], thetha[2]) * dk_uf(x_star,x_f,t_star,t_f, thetha[0], thetha[1], thetha[3])
    q = np.hstack((q_1,q_2))
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))
    f_star = q@alpha
    f_star = f_star.reshape(100,100)
    alpha_var = np.linalg.solve(L.T, np.linalg.solve(L, q.T))
    var = kernel_rbf(x_star,x_star,t_star,t_star, thetha[0], thetha[1], thetha[2]) - q@alpha_var
    var = np.diag(var)
    
    var = var.reshape(100,100)
    #std = np.sqrt(var)
    fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"})
    ax[0].plot_surface(x_star.reshape(100,100), t_star.reshape(100,100), f_star, cmap='viridis', edgecolor='none', alpha=0.5)
    ax[1].plot_surface(x_star.reshape(100,100), t_star.reshape(100,100), np.exp(-t_star.reshape(100,100))*np.sin(2*np.pi*x_star.reshape(100,100)), cmap="viridis", edgecolor='none', alpha=0.5)
    ax[0].set_title('Predictive mean')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].scatter(x_u, t_u, u_values, c='r', marker='o')
    ax[1].scatter(x_u, t_u, u_values, c='b', marker='o')
    ax[1].set_title('analytical solution')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    plt.show()
   
    fig, ax = plt.subplots()
    cont = ax.contourf(x_star.reshape(100,100), t_star.reshape(100,100), var, cmap='viridis')
    ax.scatter(x_u, t_u, c='r', marker='o')
    ax.set_title('Predictive variance')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    fig.colorbar(cont)
    

    analytical_solution = np.exp(-t_star)*np.sin(2*np.pi*x_star)
    difference_analytical_predicted_solution = np.abs(analytical_solution.reshape(100,100) - f_star)
    fig, ax = plt.subplots()
    cont = ax.contourf(x_star.reshape(100,100), t_star.reshape(100,100), difference_analytical_predicted_solution, cmap='viridis')
    plt.colorbar(cont)
    

   




#----------------------functions-----------------------------

def kernel_rbf(X,X_bar,t,t_bar, gamma_x, gamma_t, sigma_f):
        return sigma_f**2 * rbf_kernel_sklearn(X,X_bar, gamma_x) * rbf_kernel_sklearn(t,t_bar, gamma_t)

   

def dk_uf(X, X_bar, t, t_bar, gamma_x, gamma_t, alpha):
    n, m = X.shape[0], X_bar.shape[0]
    dk_uf = np.zeros((n, m))

# Loop over each pair of points
    for i in range(n):
        for j in range(m):
            diff_X = X[i] - X_bar[j]  # Squared distance between points
            diff_t = t[i] - t_bar[j]  # Difference in time
        
        # Compute dk_uf for this pair of points
            dk_uf[i, j] = -alpha * (gamma_x**4 * diff_X**2 - 2*gamma_x) + 2*gamma_t * diff_t

    return  dk_uf 
def dk_fu(X, X_bar, t, t_bar, gamma_x, gamma_t, alpha):
    n, m = X.shape[0], X_bar.shape[0]
    dk_uf = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            diff_X = X[i] - X_bar[j]  
            diff_t = (t[i] - t_bar[j])
        
        # Compute dk_uf for this pair of points
            dk_uf[i, j] = -alpha * (gamma_x**4 * diff_X**2 - 2*gamma_x) - 2*gamma_t * diff_t

    return  dk_uf 

   
def dk_ff(X, X_bar, t, t_bar, gamma_x, gamma_t, alpha):
    
    n, m = X.shape[0], X_bar.shape[0]
    dk_ff = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            
            diff_X = X[i] - X_bar[j]
            diff_t = t[i] - t_bar[j]
            #first part d/dt d/dt' 
            first_part = 2*gamma_t - 4*gamma_t**2 * diff_t**2
            #second part alpha**2 d2/dx2 d2/dx2'
            second_part = 4*gamma_x**2*(4*gamma_x*diff_X**2*(gamma_x*diff_X**2-3)+3)
            dk_ff[i, j] = first_part + alpha**2 * second_part

    return  dk_ff 

def ensure_psd(K):
    jitter = 1e-6  # Small constant
    i = 0
    while np.any(np.linalg.eigvals(K) <= 0):
        
        i += 1
        K += np.eye(K.shape[0]) * jitter
        jitter *= 10
    if i > 4:
        print(f"Attention! Added {i} times jitter to K matrix")
    return K
    
def create_K_matrix(X_u, X_f, t_u, t_f,sigma_u, sigma_f, gamma_x, gamma_t,sigma_rbf, alpha):
    k_uu = kernel_rbf(X_u,X_u,t_u,t_u,gamma_x, gamma_t, sigma_rbf ) + sigma_u * np.eye(len(X_u))
    k_uf = kernel_rbf(X_u,X_f,t_u,t_f, gamma_x, gamma_t, sigma_rbf) * dk_uf(X_u,X_f,t_u,t_f, gamma_x, gamma_t, alpha) 
    k_fu = kernel_rbf(X_f,X_u,t_f,t_u, gamma_x, gamma_t, sigma_rbf) * dk_fu(X_f,X_u,t_f,t_u, gamma_x, gamma_t, alpha)
    k_ff = kernel_rbf(X_f,X_f,t_f,t_f, gamma_x, gamma_t, sigma_rbf) * dk_ff(X_f,X_f,t_f,t_f, gamma_x, gamma_t, alpha) + sigma_f * np.eye(len(X_f))
    K = np.block([[k_uu, k_uf], [k_fu, k_ff]])
    ensure_psd(K)
    return K 


def marginal_log_likelihood(K_matrix: callable, X_u, X_f, t_u, t_f, targets, noise: list, theta: list):
    sigma_u, sigma_f = noise[0], noise[1]
    gamma_x, gamma_t, sigma_rbf = theta[0], theta[1], theta[2]
    alpha = theta[3]
    K = K_matrix(X_u, X_f, t_u, t_f,sigma_u, sigma_f, gamma_x, gamma_t,sigma_rbf, alpha)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, targets))
    marg_log_likelihood = 1/2 * np.dot(targets.T,alpha) + np.sum(np.log(np.diagonal(L))) + len(K)/2 * np.log(2*np.pi)
    return marg_log_likelihood


def grad_marg_log_likelihood(K_matrix: callable, X_u, X_f, t_u, t_f, targets, noise: list):
    def function_to_optimize(theta):
        mll = marginal_log_likelihood(K_matrix, X_u, X_f, t_u, t_f, targets, noise, theta)
        return mll
    return function_to_optimize


def optimization_restarts(K_matrix: callable,n_restarts, X_u, X_f, t_u, t_f, targets, noise: list):
    sigma_u, sigma_f = noise[0], noise[1]
    best_mll = np.inf
    best_theta = np.zeros((4))
    for i in range(n_restarts):
        rng = np.random.default_rng(seed=42)
        theta_initial = rng.uniform(0,1,4)
        res = minimize(grad_marg_log_likelihood(create_K_matrix, X_u, X_f, t_u, t_f, targets, noise), x0=theta_initial,
                    method='L-BFGS-B', bounds=((1e-5, None), (1e-5, None), (1e-5, None), (1e-5, None)))
        if res.fun < best_mll:
            best_mll = res.fun
            best_theta = res.x
    return best_theta


def rbf_kernel_second_derivative(X1, X2, gamma):
    constant = 2 * gamma
    kernel_values = rbf_kernel_sklearn(X1, X2, gamma)
    d2_kernel_fu = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[1]):
        #first part 2 * gamma * (x1_i - x2_i)**2
        factor_xx = constant * (X1[:,i] - X2[:,i])**2 - 1
        d2_kernel_fu = constant * kernel_values * factor_xx
    return d2_kernel_fu


def difference_analytical_predicted_solution():

    pass




def get_data_set(n_training_points,noise):
    
    def f(t,x):
        
        return np.exp(-t)*np.sin(2*np.pi*x) *(4*np.pi**2 - 1)
    def u(t,x):
        
        return np.exp(-t)*np.sin(2*np.pi*x)
    
    
    #alpha = 2
    # def f(t, x):
    #     return np.exp(-t) * np.sin(2 * np.pi * x) * (8 * np.pi**2 - 1)

    # def u(t, x):
    #     return np.exp(-t) * np.sin(2 * np.pi * x)


    
    # create the training data u
    rng_u = np.random.default_rng(seed=123)
    t_u = rng_u.uniform(0,1,n_training_points).reshape(-1,1)
    x_u = rng_u.uniform(0,1,n_training_points).reshape(-1,1)
    #t_u = np.sort(t_u).reshape(-1,1)
    #x_u = np.sort(x_u).reshape(-1,1)
    u_values = u(t_u,x_u).reshape(-1,1) + np.random.normal(0, noise[0], u(t_u,x_u).shape)
    
    # create the training data f
    rng_f = np.random.default_rng(seed=20)
    t_f = rng_f.uniform(0,1,n_training_points).reshape(-1,1)
    x_f = rng_f.uniform(0,1,n_training_points).reshape(-1,1)
    #t_f = np.sort(t_f).reshape(-1,1)
    #x_f = np.sort(x_f).reshape(-1,1)
    f_values = f(t_f,x_f).reshape(-1,1) + np.random.normal(0, noise[1], u(t_u,x_u).shape)

    return u_values, f_values, t_u, x_u, t_f, x_f

def plot_data_3d(u_values, f_values, t_u, x_u, t_f, x_f):
    t = np.linspace(0,1,100)
    x = np.linspace(0,1,100)
    t_mesh, x_mesh = np.meshgrid(t,x)
    u_mesh = np.exp(-t_mesh)*np.sin(2*np.pi*x_mesh)
    f_mesh = np.exp(-t_mesh)*np.sin(2*np.pi*x_mesh) *(4*np.pi**2 - 1)
    
    #plot the data in a 3d plot
    fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"})
    ax[0].plot_surface(x_mesh, t_mesh, u_mesh, cmap='viridis', edgecolor='none', alpha=0.5)
    ax[0].scatter(x_u, t_u, u_values, c='r', marker='o')
    ax[0].set_title('u(t,x)')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].set_zlabel('u(t,x)')
    ax[1].plot_surface(x_mesh, t_mesh, f_mesh, cmap='viridis', edgecolor='none', alpha=0.5)
    ax[1].scatter(x_f, t_f, f_values, c='r', marker='o')
    ax[1].set_title('f(t,x)')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    ax[1].set_zlabel('f(t,x)')
    plt.show()


def plot_data_t_0(u_values, f_values, t_u, x_u, t_f, x_f):
    #2d plot at t = 0
    plt.plot(x_u, u_values, 'ro')
    plt.plot(x_f, f_values, 'bo')
    plt.show()


if __name__ == "__main__":
    main()