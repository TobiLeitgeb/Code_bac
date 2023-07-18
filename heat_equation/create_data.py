import numpy as np
from scipy.stats import qmc
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
    engine = qmc.Sobol(d=2, scramble=True, seed=33)
    samples = engine.random(n=20)
    t_u, x_u = samples[:,0].reshape(-1,1), samples[:,1].reshape(-1,1)
    #t_u = np.sort(t_u).reshape(-1,1)
    #x_u = np.sort(x_u).reshape(-1,1)
    u_values = u(t_u,x_u).reshape(-1,1) + np.random.normal(0, noise[0], u(t_u,x_u).shape)
    
    # create the training data f
    engine = qmc.Sobol(d=2, scramble=True, seed=77)
    samples = engine.random(n=20)
    t_f, x_f = samples[:,0].reshape(-1,1), samples[:,1].reshape(-1,1)
    #t_f = np.sort(t_f).reshape(-1,1)
    #x_f = np.sort(x_f).reshape(-1,1)
    f_values = f(t_f,x_f).reshape(-1,1) + np.random.normal(0, noise[1], u(t_u,x_u).shape)
    
    return u_values, f_values, t_u, x_u, t_f, x_f