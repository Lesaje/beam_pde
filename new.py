import numpy as np
import plotly.graph_objects as go

# --- Parameters -------------------------------------------------------------
gamma = 1.0
rho1, rho2 = 1.0, 2.0
sigma1, sigma2 = 2.0, 8.0
kappa1, kappa2 = 4.0, 1.0
lambda1, lambda2 = 8.0, 4.0
L0, L = 4.0, 10.0
tau = 1.0
mu  = 0.5

# --- Discretization ---------------------------------------------------------
T = 5.0  # total time horizon
n1, n2 = 200, 300  # grid‐points on [0,L0], [L0,L]
m = 5000  # time‐steps
dt = T / m
h1 = L0 / n1
h2 = (L - L0) / n2

D = int(tau * L0 / dt)  # tau*L0 expressed in grid steps

t = np.linspace(0, T, m)
x1 = np.linspace(0, L0, n1)
x2 = np.linspace(L0, L, n2)

# --- Allocate solution arrays -----------------------------------------------
phi1 = np.zeros((n1, m))
psi1 = np.zeros((n1, m))
phi2 = np.zeros((n2, m))
psi2 = np.zeros((n2, m))
z    = np.zeros((n1, m))

# --- Nonlinearities & Loads -----------------------------------------------

f1      = lambda phi, psi: 0
f2      = lambda phi, psi: 0
g1      = lambda x: 0
h1_fun  = lambda x: 0
g2      = lambda x: 0
h2_fun  = lambda x: 0

# --- Initial Conditions -----------------------------------------------------
phi1[:, 0] = -19 / 16 * x1 ** 2 + 200 / 32 * x1
psi1[:, 0] = x1

phi2[:, 0] = 10.0 - x2
psi2[:, 0] = x2 ** 3 - (166 / 9) * x2 ** 2 + (914 / 9) * x2 - (1540 / 9)

z[:, 0] = -2*x1 + 8

phi1[:, 1] = phi1[:, 0] + dt * x1
psi1[:, 1] = psi1[:, 0] + dt * (x1 + 8)
phi2[:, 1] = phi2[:, 0] + dt * ((2 / 3) * (10.0 - x2))
psi2[:, 1] = psi2[:, 0] + dt * (2.0 * (10.0 - x2))

for j in range(1, n1):
    z[j, 1] = z[j, 0] - (dt/(tau*h1))*(z[j, 0] - z[j-1, 0]) #evaluating z(x, 1) as z(x, 0) already known

z[0,1] = (psi1[0,1] - psi1[0,0]) / dt   #z[0,t] = ψ1_t(0,t)


# --- Time stepping -----------------------------------------------------------
for n in range(1, m - 1):
    # 1) interior on I1
    for j in range(1, n1 - 1):
        phi_xx = (phi1[j + 1, n] - 2 * phi1[j, n] + phi1[j - 1, n]) / h1 ** 2
        psi_xx = (psi1[j + 1, n] - 2 * psi1[j, n] + psi1[j - 1, n]) / h1 ** 2
        psi_x = (psi1[j + 1, n] - psi1[j - 1, n]) / (2 * h1)
        phi_x = (phi1[j + 1, n] - phi1[j - 1, n]) / (2 * h1)

        phi1[j, n + 1] = (
                2 * phi1[j, n] - phi1[j, n - 1]
                + dt ** 2 / rho1 * (kappa1 * (phi_xx + psi_x)
                                    - f1(phi1[j, n], psi1[j, n])
                                    + g1(x1[j]))
        )
        psi1[j, n + 1] = (
                2 * psi1[j, n] - psi1[j, n - 1]
                + dt ** 2 / sigma1 * (lambda1 * psi_xx
                                      - kappa1 * (phi_x + psi1[j, n])
                                      - f2(phi1[j, n], psi1[j, n])
                                      + h1_fun(x1[j]))
        )

    # 2) interior on I2
    for j in range(1, n2 - 1):
        phi_xx2 = (phi2[j + 1, n] - 2 * phi2[j, n] + phi2[j - 1, n]) / h2 ** 2
        psi_xx2 = (psi2[j + 1, n] - 2 * psi2[j, n] + psi2[j - 1, n]) / h2 ** 2
        psi_x2 = (psi2[j + 1, n] - psi2[j - 1, n]) / (2 * h2)
        phi_x2 = (phi2[j + 1, n] - phi2[j - 1, n]) / (2 * h2)

        phi2[j, n + 1] = (
                2 * phi2[j, n] - phi2[j, n - 1]
                + dt ** 2 / rho2 * (kappa2 * (phi_xx2 + psi_x2)
                                    - f1(phi2[j, n], psi2[j, n])
                                    + g2(x2[j]))
        )
        psi2[j, n + 1] = (
                2 * psi2[j, n] - psi2[j, n - 1]
                + dt ** 2 / sigma2 * (lambda2 * psi_xx2
                                      - kappa2 * (phi_x2 + psi2[j, n])
                                      - f2(phi2[j, n], psi2[j, n])
                                      + h2_fun(x2[j]))
        )

    # φ1(0, t) = 0,
    phi1[0, n + 1] = 0.0
    # φ2(L, t) = 0,
    phi2[-1, n + 1] = 0.0
    # ψ2(L, t) = 0.
    psi2[-1, n + 1] = 0.0

    psi1[-1, n + 1] = (lambda2 * psi2[1, n + 1] + lambda1 * psi1[-2, n + 1]) / (lambda1 + lambda2)

    psi2[0, n + 1] = psi1[-1, n + 1]

    phi2[0, n + 1] = ((kappa2 - kappa1) * psi2[0, n+1] + kappa2 * phi2[1, n+1]/h2 + kappa1 * phi1[-2, n+1]/h2)/(kappa1/h1 + kappa2/h2)

    phi1[-1, n + 1] = phi2[0, n + 1]

    #from τ*z_t + z_x = 0, see .txt (25-44)
    for j in range(1, n1):
        z[j, n + 1] = z[j, n] - (dt / (tau * h1)) * (z[j, n] - z[j - 1, n])

    #since we already computed z(L0, n+1) we have this formula: (.txt 64)
    numer = (lambda1 / h1) * psi1[1, n + 1] + (gamma / dt) * psi1[0, n] + mu * z[-1, n + 1]
    denom = (lambda1 / h1) + (gamma / dt)
    psi1[0, n + 1] = numer / denom

    z[0, n + 1] = (psi1[0, n + 1] - psi1[0, n]) / dt  #z[0,t] = ψ1_t(0,t)



# --- Assemble full-domain solution for plotting -----------------------------
# drop the duplicate interface point at x=L0 in the second segment
X = np.concatenate((x1, x2[1:]))
Phi = np.concatenate((phi1, phi2[1:]), axis=0)
Psi = np.concatenate((psi1, psi2[1:]), axis=0)

# --- Plot with Plotly --------------------------------------------------------
figφ = go.Figure(data=[
    go.Surface(
        x=t,  # time axis
        y=X,  # spatial axis
        z=Phi,  # φ(x,t)
        colorscale='Viridis',
        showscale=True
    )
])

figφ.update_layout(
    title='Delayed‐Damped Timoshenko Transmission',
    scene=dict(
        xaxis_title='Time t',
        yaxis_title='Position x',
        zaxis_title='φ(x,t)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
    ),
    width=800,
    height=600
)

figφ.show()


figψ = go.Figure(data=[
    go.Surface(
        x=t,  # time axis
        y=X,  # spatial axis
        z=Psi,  # ψ(x,t)
        colorscale='Viridis',
        showscale=True
    )
])

figψ.update_layout(
    title='Delayed‐Damped Timoshenko Transmission',
    scene=dict(
        xaxis_title='Time t',
        yaxis_title='Position x',
        zaxis_title='ψ(x,t)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
    ),
    width=800,
    height=600
)

figψ.show()