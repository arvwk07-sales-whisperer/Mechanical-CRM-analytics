import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
L = 1.0             # Length (m)
N = 10              # Number of control volumes
k = 1.0             # Thermal conductivity
T_left = 100.0      # Boundary condition at x=0
T_right = 300.0     # Boundary condition at x=L

# Source term parameters: S(T) = -m*T + Q
m = 5.0             # heat loss coefficient
Q = 100.0           # heat generation constant

# Discretization
dx = L / N
x = np.linspace(dx/2, L - dx/2, N)

# Coefficients
aW = np.zeros(N)
aE = np.zeros(N)
aP = np.zeros(N)
b = np.zeros(N)

# Source term linearization
Sp = -m * dx   # multiplied by dx (control volume size)
Su = Q * dx

# Build coefficient matrix
for i in range(N):
    if i == 0:  # Left boundary
        aE[i] = k / dx
        aW[i] = 0.0
        aP[i] = aW[i] + aE[i] - Sp
        b[i] = aW[i]*T_left + Su
    elif i == N-1:  # Right boundary
        aW[i] = k / dx
        aE[i] = 0.0
        aP[i] = aW[i] + aE[i] - Sp
        b[i] = aE[i]*T_right + Su
    else:  # Internal nodes
        aW[i] = k / dx
        aE[i] = k / dx
        aP[i] = aW[i] + aE[i] - Sp
        b[i] = Su

# Construct matrix A
A = np.zeros((N, N))
for i in range(N):
    A[i, i] = aP[i]
    if i > 0:
        A[i, i-1] = -aW[i]
    if i < N-1:
        A[i, i+1] = -aE[i]

# Solve system
T = np.linalg.solve(A, b)

# Add boundaries
x_full = np.linspace(0, L, N+2)
T_full = np.zeros(N+2)
T_full[0] = T_left
T_full[-1] = T_right
T_full[1:-1] = T

# Plot
plt.plot(x_full, T_full, '-o', label="FVM with Source Term")
plt.xlabel("x [m]")
plt.ylabel("Temperature [°C]")
plt.title("1D Heat Conduction with Source Term Linearization")
plt.legend()
plt.grid(True)
plt.show()
