import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 41, 41   # Grid points
nt = 500          # Time steps
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
rho = 1.0         # Density
nu = 0.1          # Viscosity
dt = 0.001        # Time step

# Initialize fields
u = np.zeros((ny, nx))  # x-velocity
v = np.zeros((ny, nx))  # y-velocity
p = np.zeros((ny, nx))  # Pressure
b = np.zeros((ny, nx))  # RHS for pressure Poisson

# Pressure Poisson solver
def pressure_poisson(p, dx, dy, b, nit=50):
    for q in range(nit):
        pn = p.copy()
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,:-2])*dy**2 +
                         (pn[2:,1:-1] + pn[:-2,1:-1])*dx**2) /
                        (2*(dx**2 + dy**2)) -
                        dx**2*dy**2/(2*(dx**2+dy**2))*b[1:-1,1:-1])
        
        # Boundary conditions
        p[:,-1] = p[:,-2]   # dp/dx = 0 at x = 2
        p[:,0]  = p[:,1]    # dp/dx = 0 at x = 0
        p[-1,:] = 0         # p = 0 at y = 2
        p[0,:]  = p[1,:]    # dp/dy = 0 at y = 0
    return p

# Main time-stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Build RHS of Poisson equation
    b[1:-1,1:-1] = (rho * (1/dt * 
                  ((un[1:-1,2:] - un[1:-1,:-2])/(2*dx) +
                   (vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy)) -
                  ((un[1:-1,2:] - un[1:-1,:-2])/(2*dx))**2 -
                  2*((un[2:,1:-1] - un[:-2,1:-1])/(2*dy) *
                     (vn[1:-1,2:] - vn[1:-1,:-2])/(2*dx)) -
                  ((vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy))**2))
    
    # Pressure solve
    p = pressure_poisson(p, dx, dy, b)
    
    # Update velocity fields
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                    un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1] - un[1:-1,:-2]) -
                    vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1] - un[:-2,1:-1]) -
                    dt/(2*rho*dx)*(p[1:-1,2:] - p[1:-1,:-2]) +
                    nu*(dt/dx**2*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2]) +
                        dt/dy**2*(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])))

    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                    un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1] - vn[1:-1,:-2]) -
                    vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1] - vn[:-2,1:-1]) -
                    dt/(2*rho*dy)*(p[2:,1:-1] - p[:-2,1:-1]) +
                    nu*(dt/dx**2*(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2]) +
                        dt/dy**2*(vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1])))

    # Boundary conditions
    u[0,:] = 0
    u[:,-1] = 0
    u[:,0] = 0
    u[-1,:] = 1   # Lid moves with u=1
    v[0,:] = 0
    v[-1,:] = 0
    v[:,0] = 0
    v[:,-1] = 0

# Visualization
plt.figure(figsize=(8,6))
plt.contourf(u, cmap="jet")
plt.colorbar(label="Velocity u")
plt.title("2D Lid-Driven Cavity Flow (Navier–Stokes)")
plt.xlabel("X grid")
plt.ylabel("Y grid")
plt.show()
