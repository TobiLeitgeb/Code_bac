import pandas as pd
import numpy as np
from scipy.stats import qmc
from matplotlib import pyplot as plt
def create_data(filename, n_training_points, noise:list ):
    try:
        df= pd.read_csv("forced_oscillator/"+filename)
    except:
        df= pd.read_csv(filename)
    
    t = df['t'].values
    u = df['u'].values
    f = df['f'].values
    

    d = 1 # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
    # training data for u(t,x)
    engine = qmc.Sobol(d,seed=50)
    sample = engine.random(n_training_points)
    # sample is in [0,1]^d, so we need to scale it to the range of x and t
    indices = sample * np.array(len(t))
    indices = np.floor(indices).astype(int)
    t_train_u = t[indices]
    u_train_u = u[indices] + np.random.normal(0,noise[0],u[indices].shape)

    # training data for f(t)
    engine = qmc.Sobol(d,seed=7)
    sample = engine.random(n_training_points)
    indices = sample * np.array(len(t))
    indices = np.floor(indices).astype(int)
    t_train_f = t[indices]
    f_train_f = f[indices] + np.random.normal(0,noise[1],f[indices].shape)

    return t_train_u.reshape(-1,1), u_train_u.reshape(-1,1), t_train_f.reshape(-1,1), f_train_f.reshape(-1,1), [t,u,f]

#create_data("data.csv", 100, [0.1,0.1])

def create_validation_set(n_validation_points, noise, filename):
    try:
        df= pd.read_csv("forced_oscillator/"+filename)
    except:
        df= pd.read_csv(filename)
    
    t = df['t'].values
    u = df['u'].values
    f = df['f'].values
    


    d = 1 # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
    # training data for u(t,x)
    engine = qmc.Sobol(d,seed=50)
    sample = engine.random(n_validation_points)
    # sample is in [0,1]^d, so we need to scale it to the range of x and t
    indices = sample * np.array(len(t))
    indices = np.floor(indices).astype(int)
    t_val_u = t[indices]
    u_val_u = u[indices] + np.random.normal(0,noise[0],u[indices].shape)

    # training data for f(t)
    engine = qmc.Sobol(d,seed=7)
    sample = engine.random(n_validation_points)
    indices = sample * np.array(len(t))
    indices = np.floor(indices).astype(int)
    t_val_f = t[indices]
    f_val_f = f[indices] + np.random.normal(0,noise[1],f[indices].shape)

    return t_val_u.reshape(-1,1), u_val_u.reshape(-1,1), t_val_f.reshape(-1,1), f_val_f.reshape(-1,1)

    
