import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import qmc
# Define the grid
x = np.linspace(0, 4, 100)  # Spatial grid
t = np.linspace(0, 1, 100)  # Temporal grid

# Define the coefficients
alpha = 1  # Thermal diffusivity

# Define the initial condition
u0 = np.zeros_like(x) 
#u0 = np.ones_like(x) * 2

x = np.linspace(0, 4, 100)  # Spatial grid
t = np.linspace(0, 1, 100)  # Temporal grid
X,T = np.meshgrid(x,t)
# Define the coefficients
# alpha = 1 # Thermal diffusivity

# # Define the initial condition
# u0 = np.zeros_like(x)
# u0[1:-1] = np.sin(2*np.pi*x[1:-1])
# # Define the forcing term
# def f(x, t):
#     t = np.array(t)
#     return np.exp(-t) * np.sin(2 * np.pi * x) * (4 * np.pi**2 - 1)
# def u_analy(t,x):
#     return np.exp(-t)*np.sin(2*np.pi*x)

u0[-1] = 1
alpha = 0.001
def f(x, t):
    t = np.array(t)
    return np.where(x==0, 100*np.sin(t*np.pi), 0)


# Define the finite difference operator
diagonals = [-2 * np.ones(100), np.ones(99), np.ones(99)]
laplacian = diags(diagonals, [0, -1, 1]).toarray() / (x[1] - x[0])**2

# Define the ODE system (the heat equation in this case)
def heat_eqn(t, u):
    return alpha * np.dot(laplacian, u) + f(x, t)

# Solve the ODE system
sol = solve_ivp(heat_eqn, (0, 1), u0, method='RK45', t_eval=t)
print(sol.y.shape)
solution_x = sol.y[:,0].reshape(-1,1)
solution_t = sol.y[0,:].reshape(-1,1)
#change axis
sol = sol.y.T
#now the t axis is in the wrong direction

# Define the size of your grid
n_rows = 100
n_cols = 100
def u(t,x):
        return np.exp(-t)*np.sin(2*np.pi*x)
#

fig, ax = plt.subplots(1,2,figsize=(20,20),subplot_kw={"projection": "3d"})
ax[0].plot_surface(X, T, sol, cmap='viridis',alpha=0.8)
#ax[0].plot_surface(X, T, u(T,X),alpha=0.6)

#ax[0].plot_surface(X, T, u(T,X),alpha=0.6)
ax[0].set_xlabel('x')
ax[0].set_ylabel('t')
ax[0].set_title('Numerical solution')
ax[1].plot_surface(X, T, f(X,T), cmap='viridis')
ax[1].set_xlabel('x')

ax[1].set_ylabel('t')
ax[1].set_title('focring term')
plt.show()

