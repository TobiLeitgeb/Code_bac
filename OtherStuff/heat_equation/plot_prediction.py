import matplotlib.pyplot as plt
import numpy as np
from create_data import f,u

def plot_prediction(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, mean_u, mean_f, title:str, save_path:str):
    """plots the prediction of the model"""
    fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"},figsize=(12,5))

    ax[0].plot_surface(x_star.reshape(mean_u.shape), t_star.reshape(mean_u.shape), mean_u, cmap='viridis', edgecolor='none', alpha=0.5)
    ax[0].set_title('Predictive mean for u(t,x))')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].scatter(x_u, t_u, u_train, c='r', marker='o')

    ax[1].plot_surface(x_star.reshape(mean_f.shape), t_star.reshape(mean_f.shape), mean_f, cmap="viridis", edgecolor='none', alpha=0.5)
    ax[1].scatter(x_f, t_f, f_train, c='b', marker='o')
    ax[1].set_title('Predictive mean for f(t,x)')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    plt.suptitle(title)
    plt.savefig("heat_equation/"+save_path)

def plot_variance(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, var_u, var_f, title:str, save_path:str):
    """plots the variance of the model"""
    fig, ax = plt.subplots(1,2,figsize=(12,5))

    cont_0 = ax[0].contourf(x_star.reshape(var_u.shape), t_star.reshape(var_u.shape), np.sqrt(var_u), cmap='viridis', alpha=0.8)
    ax[0].set_title('Predictive std for u(t,x))')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].scatter(x_u, t_u, marker='o')
    fig.colorbar(cont_0, ax=ax[0])

    cont_1 = ax[1].contourf(x_star.reshape(var_f.shape), t_star.reshape(var_f.shape), np.sqrt(var_f), cmap="viridis", alpha=0.8)
    ax[1].scatter(x_f, t_f, marker='o')
    ax[1].set_title('Predictive std for f(t,x)')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    fig.colorbar(cont_1, ax=ax[1])

    plt.suptitle(title)
    plt.savefig("heat_equation/"+save_path)

def plot_difference_analytical(x_u, t_u, x_f, t_f, u_train, f_train, x_star, t_star, mean_u, mean_f, title:str, save_path:str):
    """plots the difference between the analytical solution and the predicted mean"""
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    if None in mean_u:
        print('achtung')
    cont = ax[0].contourf(x_star.reshape(mean_u.shape), t_star.reshape(mean_u.shape), np.abs(mean_u - u(t_star,x_star).reshape(mean_u.shape)), cmap='viridis', alpha=0.8)
    ax[0].set_title(' $f_*$ - u(t,x))')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].scatter(x_u, t_u, c='r', marker='o')
    fig.colorbar(cont, ax=ax[0])

    cont2 = ax[1].contourf(x_star.reshape(mean_f.shape), t_star.reshape(mean_f.shape), np.abs(mean_f - f(t_star,x_star).reshape(mean_u.shape)), cmap="viridis", alpha=0.8)
    ax[1].scatter(x_f, t_f, c='b', marker='o')
    ax[1].set_title(' $f_*$ - f(t,x)')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    fig.colorbar(cont2, ax=ax[1])
    plt.suptitle(title)
    plt.savefig("heat_equation/"+save_path)