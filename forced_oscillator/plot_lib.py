import matplotlib.pyplot as plt
import numpy as np

def plot_prediction (t_u,Y_u,t_f,Y_f, x_star, f_star_1, var_1,f_star_2, var_2, title:str, savefig:bool=False, filename:str=None):
    """plots the prediction of the model"""
    fig,ax = plt.subplots(1,2,figsize=(12, 6))
    ax[0].plot(t_u, Y_u, 'r.', markersize=10, label='Observations')
    ax[0].plot(x_star, f_star_1, 'b', label='Prediction')
    ax[0].fill_between(x_star.flatten(), f_star_1.flatten() - 2 * np.sqrt(var_1.flatten()),
                     f_star_1.flatten() + 2 * np.sqrt(var_1.flatten()), alpha=0.2, color='blue')
    ax[0].legend(loc='upper left', fontsize=12)
    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('$u(t)$')
    ax[0].set_title("u(t) prediction")
    ax[0].grid(alpha = 0.7)
    ax[1].plot(t_f, Y_f, 'r.', markersize=10, label='Observations')
    ax[1].plot(x_star, f_star_2, 'b', label='Prediction')
    ax[1].fill_between(x_star.flatten(), f_star_2.flatten() - 2 * np.sqrt(var_2.flatten()),
                        f_star_2.flatten() + 2 * np.sqrt(var_2.flatten()), alpha=0.2, color='blue')
    ax[1].legend(loc='upper left', fontsize=12)
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$f(t)$')
    ax[1].set_title("f(t) prediction")
    ax[1].grid(alpha = 0.7)
    fig.suptitle(title)
    
    return fig,ax