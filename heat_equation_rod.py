import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Define the grid
x = np.linspace(0, 1, 100)  # Spatial grid
t = np.linspace(0, 1, 100)  # Temporal grid

# Define the coefficients
alpha = 0.1  # Thermal diffusivity

# Define the initial condition
u0 = np.zeros_like(x) 


# Define the forcing term
def f(t, x):
    # Forcing term here is just a simple example. You can replace with your own function.
    if t < 0.3:
        return np.where(np.logical_and(x >= 0.2, x <= 0.4), 0.1, 0) + np.where(np.logical_and(x >= 0.6, x <= 0.8), np.sin(t*2), 0)
    else:
        return 0
      

# Define the finite difference operator
diagonals = [-2 * np.ones(100), np.ones(99), np.ones(99)]
laplacian = diags(diagonals, [0, -1, 1]).toarray() / (x[1] - x[0])**2

# Define the ODE system (the heat equation in this case)
def heat_eqn(t, u):
    return alpha * np.dot(laplacian, u) + f(t, x)

# Solve the ODE system
sol = solve_ivp(heat_eqn, (0, 1), u0, method='RK45', t_eval=t)
sol.y.shape

# Plot the solution as 3d plot
X, T = np.meshgrid(x, t)
# def f(t,x):
#         return np.exp(-t)*np.sin(2*np.pi*x) *(4*np.pi**2 - 1)
# def u(t,x):
#     return np.exp(-t)*np.sin(2*np.pi*x)
def f(t, x):
    # Forcing term here is just a simple example. You can replace with your own function.
    
    return np.where(np.logical_and(x >= 0.2, x <= 0.4,t<0.3), 0.5, 0) 
    
fig, ax = plt.subplots(2,1,subplot_kw={"projection": "3d"})
ax[0].plot_surface(T, X, sol.y, cmap='viridis')
ax[0].set_xlabel('x')
ax[0].set_ylabel('t')
ax[0].set_title('Numerical solution')
ax[1].plot_surface(T, X, f(T,X), cmap='viridis')
ax[1].set_xlabel('x')
ax[1].set_ylabel('t')
ax[1].set_title('focring term')
plt.show()