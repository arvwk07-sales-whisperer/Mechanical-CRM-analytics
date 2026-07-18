import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 41, 41
nt = 500           # time steps
dx = dy = 2/(nx-1)
alpha = 0.1        # thermal diffusivity
dt = 0.001

# Initialize temperature field
T = np.zeros((ny, nx))

# Boundary condition: left wall hot
T[:, 0] = 100  

# Time stepping
for n in range(nt):
    Tn = T.copy()
    T[1:-1,1:-1] = (Tn[1:-1,1:-1] +
                     alpha*dt/dx**2*(Tn[1:-1,2:] - 2*Tn[1:-1,1:-1] + Tn[1:-1,:-2]) +
                     alpha*dt/dy**2*(Tn[2:,1:-1] - 2*Tn[1:-1,1:-1] + Tn[:-2,1:-1]))

    # Boundary conditions
    T[:,0]  = 100   # Left wall hot
    T[:,-1] = 0     # Right wall cold
    T[0,:]  = 0     # Bottom wall cold
    T[-1,:] = 0     # Top wall cold

# Plot
plt.figure(figsize=(6,5))
plt.contourf(T, cmap="inferno")
plt.colorbar(label="Temperature [°C]")
plt.title("2D Heat Conduction (Transient Diffusion)")
plt.xlabel("X grid")
plt.ylabel("Y grid")
plt.show()
