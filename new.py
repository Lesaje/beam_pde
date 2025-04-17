import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
gamma, rho1, rho2, sigma1, sigma2 = 1, 1, 2, 2, 1
kappa1, kappa2, lambda1, lambda2 = 4, 1, 8, 4
L, L0 = 10, 4
mu, tau_delay = 0, 0

# Discretization
n1, n2, m = 20, 20, 500
h1, h2 = L0 / (n1 - 1), (L - L0) / (n2 - 1)
T = 10
tau = T / (m - 1)

# Grids
x1 = np.linspace(0, L0, n1)
x2 = np.linspace(L0, L, n2)
t = np.linspace(0, T, m)

# Initialize arrays
phi1, psi1 = np.zeros((n1, m)), np.zeros((n1, m))
phi2, psi2 = np.zeros((n2, m)), np.zeros((n2, m))

# Initial conditions (set to zero initially for simplicity)

# Functions
f1 = lambda phi, psi: 4 * (phi + psi) ** 3 - 2 * (phi + psi) + 2 * phi * psi ** 2
f2 = lambda phi, psi: 4 * (phi + psi) ** 3 - 2 * (phi + psi) + 2 * phi ** 2 * psi
g1, h1_fun = lambda x: np.sin(x), lambda x: x
g2, h2_fun = lambda x: np.cos(x), lambda x: x + 1

# Time
for n in range(1, m - 1):
    # Interior points
    for j in range(1, n1 - 1):
        phi1_xx = (phi1[j + 1, n] - 2 * phi1[j, n] + phi1[j - 1, n]) / h1 ** 2
        psi1_xx = (psi1[j + 1, n] - 2 * psi1[j, n] + psi1[j - 1, n]) / h1 ** 2
        phi1_xt = (phi1[j + 1, n] - phi1[j - 1, n]) / (2 * h1)

        phi1[j, n + 1] = (2 * phi1[j, n] - phi1[j, n - 1] + tau ** 2 / rho1 * (
                kappa1 * (phi1_xx + psi1_xx) - f1(phi1[j, n], psi1[j, n]) + g1(x1[j])))
        psi1[j, n + 1] = (2 * psi1[j, n] - psi1[j, n - 1] + tau ** 2 / sigma1 * (
                lambda1 * psi1_xx - kappa1 * (phi1_xt + psi1[j, n]) - f2(phi1[j, n], psi1[j, n]) + h1_fun(x1[j])))

    # Boundary conditions at x=0 for phi1 and psi1
    phi1[0, n + 1] = 0
    psi1[0, n + 1] = (lambda1 * psi1[1, n + 1] / h1) / (lambda1 / h1 + gamma / tau)

    # Interior points for phi2, psi2
    for j in range(1, n2 - 1):
        phi2_xx = (phi2[j + 1, n] - 2 * phi2[j, n] + phi2[j - 1, n]) / h2 ** 2
        psi2_xx = (psi2[j + 1, n] - 2 * psi2[j, n] + psi2[j - 1, n]) / h2 ** 2
        phi2_xt = (phi2[j + 1, n] - phi2[j - 1, n]) / (2 * h2)

        phi2[j, n + 1] = (2 * phi2[j, n] - phi2[j, n - 1] + tau ** 2 / rho2 * (
                kappa2 * (phi2_xx + psi2_xx) - f1(phi2[j, n], psi2[j, n]) + g2(x2[j])))
        psi2[j, n + 1] = (2 * psi2[j, n] - psi2[j, n - 1] + tau ** 2 / sigma2 * (
                lambda2 * psi2_xx - kappa2 * (phi2_xt + psi2[j, n]) - f2(phi2[j, n], psi2[j, n]) + h2_fun(x2[j])))

    # Boundary conditions at x=L for phi2, psi2[0]
    phi2[-1, n + 1] = 0
    psi2[0, n + 1] = 0

    # Transmission conditions at x=L0
    phi1[-1, n + 1] = phi2[0, n + 1] = (phi1[-2, n + 1] + phi2[1, n + 1]) / 2
    psi1[-1, n + 1] = psi2[0, n + 1] = (psi1[-2, n + 1] + psi2[1, n + 1]) / 2

# Visualization
X = np.concatenate((x1, x2))
PHI = np.vstack((phi1, phi2))
PSI = np.vstack((psi1, psi2))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
T_mesh, X_mesh = np.meshgrid(t, X)
ax.plot_surface(T_mesh, X_mesh, PHI, cmap='viridis')
ax.set_xlabel('Time (t)')
ax.set_ylabel('Position (x)')
ax.set_zlabel('Displacement (phi)')
plt.title('Timoshenko Beam Transmission Problem Solution')
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
T_mesh, X_mesh = np.meshgrid(t, X)
ax.plot_surface(T_mesh, X_mesh, PSI, cmap='plasma')
ax.set_xlabel('Time (t)')
ax.set_ylabel('Position (x)')
ax.set_zlabel('Shear angle (psi)')
plt.title('Timoshenko Beam Shear Angle Variation')
plt.show()