import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
L = 1.0             # Length of the domain (m)
N = 10              # Number of control volumes
k = 1.0             # Thermal conductivity (W/m-K)
T_left = 100.0      # Boundary condition at x=0
T_right = 300.0     # Boundary condition at x=L

# Discretization
dx = L / N
x = np.linspace(dx/2, L - dx/2, N)  # Cell centers

# Coefficient arrays
aW = np.zeros(N)
aE = np.zeros(N)
aP = np.zeros(N)
b = np.zeros(N)

# Assemble the coefficient matrix
# Assemble the coefficient matrix
for i in range(N):
    if i == 0:  # Left boundary
        aE[i] = k / dx
        aW[i] = 0.0
        aP[i] = aW[i] + aE[i] + (2 * k / dx)
        b[i] = (2 * k / dx) * T_left
    elif i == N-1:  # Right boundary
        aE[i] = 0.0
        aW[i] = k / dx
        aP[i] = aW[i] + aE[i] + (2 * k / dx)
        b[i] = (2 * k / dx) * T_right
    else:  # Internal nodes
        aE[i] = k / dx
        aW[i] = k / dx
        aP[i] = aW[i] + aE[i]
        b[i] = 0.0

# Solve the system of equations
T = np.linalg.solve(np.diag(aP) - np.diag(aE[:-1], 1) - np.diag(aW[1:], -1), b)

# Plot the results
plt.plot(x, T, 'o-', label='Numerical Solution')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (C)')
plt.title('1D Steady-State Heat Conduction')
plt.grid(True)
plt.legend()
plt.show()
