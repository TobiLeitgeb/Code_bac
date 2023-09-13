import numpy as np
from scipy.stats import qmc
from create_data import u,f
from posterior import posterior_distribution_u, posterior_distribution_f
from sklearn.metrics import mean_squared_error



def mse(x_u, x_f, t_u, t_f,targets,n_test_points, theta, noise, gram_Matrix):
    """returns the mean squared error of the model"""
    
    engine = qmc.Sobol(d=2, scramble=True, seed=78)
    samples = engine.random(n=n_test_points)
    t_test, x_test = samples[:,0].reshape(-1,1), samples[:,1].reshape(-1,1)
    t_test, x_test = np.meshgrid(t_test, x_test)
    u_test = u(t_u,x_u).reshape(-1,1) + np.random.normal(0, noise[0], u(t_u,x_u).shape)
    f_test = f(t_f,x_f).reshape(-1,1) + np.random.normal(0, noise[1], u(t_u,x_u).shape)
    print('shapes',u_test.shape, f_test.shape)
    mean_u, _ = posterior_distribution_u(x_u, x_f, t_u, t_f, x_test, t_test, targets, noise, theta, gram_Matrix)
    mean_f, _ = posterior_distribution_f(x_u, x_f, t_u, t_f, x_test, t_test, targets, noise, theta, gram_Matrix)
    mean_u, mean_f = mean_u.reshape(-1,1),mean_f.reshape(-1,1)
    print('shapes',mean_u.shape, mean_f.shape)
    return mean_squared_error(u_test, mean_u), mean_squared_error(f_test, mean_f)