import numpy as np
from scipy.stats import qmc
from scipy.sparse import diags
from scipy.integrate import solve_ivp



def f(t,x):
        return np.exp(-t)*np.sin(2*np.pi*x) *(4*np.pi**2 - 1)
def u(t,x):
    return np.exp(-t)*np.sin(2*np.pi*x)
def get_data_set(n_training_points,noise):
    
    
    #alpha = 2
    # def f(t, x):
    #     return np.exp(-t) * np.sin(2 * np.pi * x) * (8 * np.pi**2 - 1)

    # def u(t, x):
    #     return np.exp(-t) * np.sin(2 * np.pi * x)

    # create the training data u
    engine = qmc.Sobol(d=2, scramble=True, seed=5)
    samples = engine.random(n=n_training_points)
    t_u, x_u = samples[:,0].reshape(-1,1), samples[:,1].reshape(-1,1)
    u_values = u(t_u,x_u).reshape(-1,1) + np.random.normal(0, noise[0], u(t_u,x_u).shape)
    
    # create the training data f
    engine = qmc.Sobol(d=2, scramble=True, seed=75)
    samples = engine.random(n=n_training_points)
    t_f, x_f = samples[:,0].reshape(-1,1), samples[:,1].reshape(-1,1)
    f_values = f(t_f,x_f).reshape(-1,1) + np.random.normal(0, noise[1], u(t_u,x_u).shape)
    
    # create the test data
   
    return u_values, f_values, t_u, x_u, t_f, x_f

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import pandas as pd



def get_data_set2(n_training_points, noise,filename):
    #load data from mathematica calculation
    try:
        df= pd.read_csv("heat_equation/"+filename)
    except:
        df= pd.read_csv(filename)
    x = df['space'].values
    t = df['time'].values
    u = df['u'].values
    f = df['f'].values

    x_mesh = x.reshape(int(np.sqrt(len(x))),int(np.sqrt(len(x))))
    t_mesh = t.reshape(x_mesh.shape)
    u_grid = u.reshape(x_mesh.shape)
    f_grid = f.reshape(x_mesh.shape)


    
    # 
    x_axis = np.unique(x_mesh)
    t_axis = np.unique(t_mesh)
    
    
    d = 2 # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
    # training data for u(t,x)
    engine = qmc.Sobol(d,seed=77)
    sample = engine.random(n_training_points)
    # sample is in [0,1]^d, so we need to scale it to the range of x and t
    indices = sample * np.array([len(x_axis),len(t_axis)])
    indices = np.floor(indices).astype(int)
    x_train_u = x_axis[indices[:,0]]
    t_train_u = t_axis[indices[:,1]]
    u_train = u_grid[indices[:,1],indices[:,0]] + np.random.normal(0, noise[0], u_grid[indices[:,1],indices[:,0]].shape)
    
    #same thing for f(t,x)
    engine = qmc.Sobol(d,seed=100)
    sample = engine.random(n_training_points)
    indices = sample * np.array([len(x_axis),len(t_axis)])
    indices = np.floor(indices).astype(int)
    x_train_f = x_axis[indices[:,0]]
    t_train_f = t_axis[indices[:,1]]
    f_train = f_grid[indices[:,1],indices[:,0]] + np.random.normal(0, noise[1], f_grid[indices[:,1],indices[:,0]].shape)

    


    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].contourf(x_mesh,t_mesh,u_grid,cmap='viridis',alpha=0.8)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].set_title('u(t,x)')
    ax[0].scatter(x_train_u,t_train_u,c='r',marker='o')
    ax[1].contourf(x_mesh,t_mesh,f_grid,cmap='viridis',edgecolor='none',alpha=0.8)
    ax[1].scatter(x_train_f,t_train_f,c='r',marker='o')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    ax[1].set_title('f(t,x)')
    #plt.show()
    return x_train_u.reshape(-1,1), x_train_f.reshape(-1,1), t_train_u.reshape(-1,1), t_train_f.reshape(-1,1), u_train.reshape(-1,1), f_train.reshape(-1,1), [x_mesh,t_mesh,u_grid,f_grid]
    

def create_validation_set(n_validation_points, noise, filename):
    try:
        df= pd.read_csv("heat_equation/"+filename)
    except:
        df= pd.read_csv(filename)
    x = df['space'].values
    t = df['time'].values
    u = df['u'].values
    f = df['f'].values

    x_mesh = x.reshape(int(np.sqrt(len(x))),int(np.sqrt(len(x))))
    t_mesh = t.reshape(x_mesh.shape)
    u_grid = u.reshape(x_mesh.shape)
    f_grid = f.reshape(x_mesh.shape)

    x_axis = np.unique(x_mesh)
    t_axis = np.unique(t_mesh)

    d = 2 # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
    # training data for u(t,x)
    engine = qmc.Sobol(d)
    sample = engine.random_base2(n_validation_points)
    # sample is in [0,1]^d, so we need to scale it to the range of x and t
    indices = sample * np.array([len(x_axis),len(t_axis)])
    indices = np.floor(indices).astype(int)
    x_val_u = x_axis[indices[:,0]]
    t_val_u = t_axis[indices[:,1]]
    u_val = u_grid[indices[:,1],indices[:,0]] + np.random.normal(0, noise[0], u_grid[indices[:,1],indices[:,0]].shape)
    
    #same thing for f(t,x)
    engine = qmc.Sobol(d)
    sample = engine.random_base2(n_validation_points)
    indices = sample * np.array([len(x_axis),len(t_axis)])
    indices = np.floor(indices).astype(int)
    x_val_f = x_axis[indices[:,0]]
    t_val_f = t_axis[indices[:,1]]
    f_val = f_grid[indices[:,1],indices[:,0]] + np.random.normal(0, noise[1], f_grid[indices[:,1],indices[:,0]].shape)

    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].contourf(x_mesh,t_mesh,u_grid,cmap='viridis',alpha=0.8)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].set_title(f'Validation set for u(t,x), n = {2**n_validation_points}')
    ax[0].scatter(x_val_u,t_val_u,c='r',marker='x')
    ax[1].contourf(x_mesh,t_mesh,f_grid,cmap='viridis',alpha=0.8)
    ax[1].scatter(x_val_f,t_val_f,c='r',marker='x')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    ax[1].set_title(f'validation set for f(t,x), n = {2**n_validation_points}')
    plt.show()

    return [x_val_u, x_val_f, t_val_u, t_val_f, u_val, f_val]
