import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Parameters -------------------------------------------------------------
gamma     = 1.0
rho1, rho2    = 1.0, 2.0
sigma1, sigma2 = 2.0, 1.0
kappa1, kappa2 = 4.0, 1.0
lambda1, lambda2 = 8.0, 4.0
L0, L      = 4.0, 10.0
mu        = 0.5          # delay feedback coefficient
tau_delay  = 1.0         # transport‐delay PDE parameter

# Add small linear damping to stabilize nonlinear blow-up
beta_phi1, beta_psi1 = 0.01, 0.01
beta_phi2, beta_psi2 = 0.01, 0.01

# --- Discretization ---------------------------------------------------------
T         = 10.0        # total time horizon
n1, n2    = 500, 500      # grid points in I1=[0,L0], I2=[L0,L]
m          = 20000       # time steps
dt         = T / (m - 1)
h1         = L0 / (n1 - 1)
h2         = (L - L0) / (n2 - 1)

t         = np.linspace(0, T, m)
x1        = np.linspace(0, L0, n1)
x2        = np.linspace(L0, L, n2)

delay_steps = max(1, int((tau_delay * L0) / dt))

# --- Allocate solution arrays -----------------------------------------------
phi1 = np.zeros((n1, m))   # phi on I1
psi1 = np.zeros((n1, m))   # psi on I1
phi2 = np.zeros((n2, m))   # phi on I2
psi2 = np.zeros((n2, m))   # psi on I2
z    = np.zeros((n1, m))   # delay‐PDE on I1

# --- Right‐hand sides & nonlinearities --------------------------------------
g1      = lambda x: np.sin(x)
h1_fun  = lambda x: x
g2      = lambda x: np.cos(x)
h2_fun  = lambda x: x + 1
f1      = lambda phi, psi: 4*(phi + psi)**3 - 2*(phi + psi) + 2*phi*psi**2
f2      = lambda phi, psi: 4*(phi + psi)**3 - 2*(phi + psi) + 2*phi**2*psi

# --- Initial conditions -----------------------------------------------------
phi1[:,0] = -19/16*x1**2 + 200/32*x1
psi1[:,0] = x1
phi1_L0    = phi1[-1,0]
# choose phi2^0(x) continuous: linear from phi1^0(L0) down to zero at x=L
phi2[:,0] = phi1_L0 * (L - x2) / (L - L0)
psi2[:,0] = x2**3 - (166/9)*x2**2 + (914/9)*x2 - (1540/9)

# first step using given initial velocities
dphi1_0   = x1
dpsi1_0   = 10.0 - x1
dphi2_0   = (2/3)*(10 - x2)
dpsi2_0   = (5/6)*(10 - x2)
phi1[:,1] = phi1[:,0] + dt * dphi1_0
psi1[:,1] = psi1[:,0] + dt * dpsi1_0
phi2[:,1] = phi2[:,0] + dt * dphi2_0
psi2[:,1] = psi2[:,0] + dt * dpsi2_0

# delay‐PDE z initial condition and first step
z[:,0] = 13/4 * x1 + 1
for j in range(1, n1):
    z[j,1] = z[j,0] - dt/(tau_delay*h1)*(z[j,0] - z[j-1,0])
# z boundary at t=dt will be set in loop

# --- Time stepping -----------------------------------------------------------
for n in range(1, m-1):
    # interior update on I1
    for j in range(1, n1-1):
        # spatial derivatives
        phi1_xx = (phi1[j+1,n] - 2*phi1[j,n] + phi1[j-1,n]) / h1**2
        psi1_xx = (psi1[j+1,n] - 2*psi1[j,n] + psi1[j-1,n]) / h1**2
        psi1_x  = (psi1[j+1,n] - psi1[j-1,n]) / (2*h1)
        # time derivatives
        phi1_t  = (phi1[j,n] - phi1[j,n-1]) / dt
        psi1_t  = (psi1[j,n] - psi1[j,n-1]) / dt
        # update with linear damping
        phi1[j,n+1] = (2*phi1[j,n] - phi1[j,n-1]
                       - beta_phi1*dt*phi1_t
                       + dt**2/rho1*(kappa1*(phi1_xx + psi1_x)
                                    - f1(phi1[j,n], psi1[j,n])
                                    + g1(x1[j])))
        phi1_x   = (phi1[j+1,n] - phi1[j-1,n]) / (2*h1)
        psi1[j,n+1] = (2*psi1[j,n] - psi1[j,n-1]
                       - beta_psi1*dt*psi1_t
                       + dt**2/sigma1*(lambda1*psi1_xx
                                      - kappa1*(phi1_x + psi1[j,n])
                                      - f2(phi1[j,n], psi1[j,n])
                                      + h1_fun(x1[j])))

    # interior update on I2
    for j in range(1, n2-1):
        phi2_xx = (phi2[j+1,n] - 2*phi2[j,n] + phi2[j-1,n]) / h2**2
        psi2_xx = (psi2[j+1,n] - 2*psi2[j,n] + psi2[j-1,n]) / h2**2
        psi2_x  = (psi2[j+1,n] - psi2[j-1,n]) / (2*h2)
        phi2_t  = (phi2[j,n] - phi2[j,n-1]) / dt
        psi2_t  = (psi2[j,n] - psi2[j,n-1]) / dt
        phi2[j,n+1] = (2*phi2[j,n] - phi2[j,n-1]
                       - beta_phi2*dt*phi2_t
                       + dt**2/rho2*(kappa2*(phi2_xx + psi2_x)
                                    - f1(phi2[j,n], psi2[j,n])
                                    + g2(x2[j])))
        phi2_x   = (phi2[j+1,n] - phi2[j-1,n]) / (2*h2)
        psi2[j,n+1] = (2*psi2[j,n] - psi2[j,n-1]
                       - beta_psi2*dt*psi2_t
                       + dt**2/sigma2*(lambda2*psi2_xx
                                      - kappa2*(phi2_x + psi2[j,n])
                                      - f2(phi2[j,n], psi2[j,n])
                                      + h2_fun(x2[j])))

    # upwind‐transport for z interior
    for j in range(1, n1):
        z[j,n+1] = z[j,n] - dt/(tau_delay*h1)*(z[j,n] - z[j-1,n])

    # boundary conditions on I1 at x=0
    phi1[0,n+1] = 0.0
    lhs = lambda1 / h1

    # boundary conditions on I2 at x=L and transmission at x=L0
    phi2[-1,n+1] = 0.0
    psi2[0,n+1] = 0.0
    phi_int = 0.5*(phi1[-2,n+1] + phi2[1,n+1])
    psi_int = 0.5*(psi1[-2,n+1] + psi2[1,n+1])
    phi1[-1,n+1] = phi2[0,n+1] = phi_int
    psi1[-1,n+1] = psi2[0,n+1] = psi_int

    # delayed‐feedback boundary for psi1
    # compute psi1_t0 safely
    psi1_t0 = (psi1[0,n] - psi1[0,n-1]) / dt
    if not np.isfinite(psi1_t0):
        psi1_t0 = 0.0
    z[0,n] = psi1_t0

    # compute delayed boundary derivative safely
    idx_d = n - delay_steps
    if idx_d >= 1:
        delayed_derivative = (psi1[0,idx_d] - psi1[0,idx_d-1]) / dt
        if not np.isfinite(delayed_derivative):
            delayed_derivative = 0.0
        z[-1,n] = delayed_derivative
    else:
        z[-1,n] = 13/4 * x1[-1] + 1

    # combine to form rhs, guarding against NaN/inf
    rhs = gamma * z[0,n] + mu * z[-1,n]
    if not np.isfinite(rhs):
        rhs = 0.0

    # apply boundary condition
    psi1[0,n] = (lhs*psi1[1,n] - rhs) / lhs

# --- Visualization -----------------------------------------------------------
X   = np.concatenate((x1, x2))
Phi = np.vstack((phi1, phi2))
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
T_mesh, X_mesh = np.meshgrid(t, X)
ax.plot_surface(T_mesh, X_mesh, Phi, rstride=10, cstride=10, cmap='viridis')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('phi(x,t)')
plt.title('Stabilized Delayed Timoshenko Transmission Solution')
plt.tight_layout()
plt.show()

