import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

al1 = 1
be1 = 1
al2 = 1
be2 = 4
ga = 1
a = -5 / 618
c = 469 / 1236
k1 = 25 / 88992
l = -145 / 22248
q = 25 / 927
r = 181 / 11124
p = 2360 / 2781
a1 = -5 / 48
b = 2 / 3
d = -1 / 6
g = 5 / 3
L = 10
L0 = 4
T = 10
n1 = 5
n2 = 7
m = 300
l1 = L0 / (n1 - 1)
l2 = (L - L0) / (n2 - 1)
tau = T / (m - 1)


def v0(x):
    return k1 * x ** 4 + l * x ** 3 + q * x ** 2 + r * x + p


def u0(x):
    return a * x ** 3 + c * x


def u1(x):
    return a1 * x ** 2 + b * x


def v1(x):
    return d * x + g


def f1(u, ux):
    return 4 * (u + ux) ** 3 - 2 * (u + ux) + 2 * u * ux ** 2


def f2(v, vx):
    return 4 * (v + vx) ** 3 - 2 * (v + vx) + 2 * v * vx ** 2


def g1(u, ux):
    return 4 * (u + ux) ** 3 - 2 * (u + ux) + 2 * ux * u ** 2


def g2(v, vx):
    return 4 * (v + vx) ** 3 - 2 * (v + vx) + 2 * vx * v ** 2


def g1x(u, ux, uxx):
    return 12 * (u - ux) ** 2 * (ux - uxx) - 2 * (ux - uxx) + 2 * uxx * u ** 2 + 4 * ux ** 2


def g2x(v, vx, vxx):
    return 12 * (v - vx) ** 2 * (vx - vxx) - 2 * (vx - vxx) + 2 * vxx * v ** 2 + 4 * vx ** 2


u = np.zeros([n1, m])
v = np.zeros([n2, m])

t = np.zeros(m)
t[0] = 0
for j in range(0, m):
    t[j] = tau * j

X1 = np.zeros(n1)
X1[0] = 0
for i in range(0, n1):
    X1[i] = (L0 + l1) * i / n1

X2 = np.zeros(n2)
X2[0] = 0
for i in range(0, n2):
    X2[i] = L0 + (L - L0 + l2) * i / n2

X = np.hstack((X1, X2))
for j in range(0, m - 1):
    u[0, j] = 0
    v[n2 - 1, j] = 0

for i in range(0, n1):
    u[i, 0] = u0(X1[i])
    u[i, 1] = u1(X1[i]) * tau + u[i, 0]

for k in range(0, n2):
    v[k, 0] = v0(X2[k])
    v[k, 1] = v1(X2[k]) * tau + v[k, 0]

h = np.zeros([3, 3])
f = np.zeros(3)
h[0, 0] = -1 / l1
h[1, 0] = -2 / l1 ** 2
h[2, 0] = -3 / l1 ** 3
h[0, 1] = 1 / l1 + 1 / l2
h[1, 1] = 1 / l1 ** 2 - be2 / l2 ** 2
h[2, 1] = 1 / l1 ** 3 + be2 / l2 ** 3
h[0, 2] = -1 / l2
h[1, 2] = 2 * be2 / l2 ** 2
h[2, 2] = -3 * be2 / l2 ** 3
f[0] = 0
# - g1x(u[i, j], (u[i+1, j] - u[i, j])/l1, (u[i+1, j] - 2*u[i, j] + u[i-1, j])/l1**2)
# - g2x(v[k, j], (v[k+1, j] - v[k, j])/l2, (v[k+1, j] - 2*v[k, j] + v[k-1, j])/l2**2)
for j in range(1, m - 1):
    for i in range(2, n1 - 2):
        u[i, j + 1] = -((u[i + 2, j] - 4 * u[i + 1, j] + 6 * u[i, j] - 4 * u[i - 1, j] + u[i - 2, j]) / l1 ** 4 + f1(
            u[i, j], - (u[i, j] - u[i - 1, j]) / l1) - g1x(u[i, j], (u[i + 1, j] - u[i, j]) / l1,
                                                           (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / l1 ** 2) - u[
                            i, j] / tau + (- 2 * u[i, j] + u[i, j - 1]) / tau ** 2) / (al1 / tau ** 2 + ga / tau)
        for k in range(2, n2 - 2):
            v[k, j + 1] = -4 * tau ** 2 * (
                        be2 * (v[k + 2, j] - 4 * v[k + 1, j] + 6 * v[k, j] - 4 * v[k - 1, j] + v[k - 2, j]) / (
                            9 * l2 ** 4) + f2(v[k, j], - (v[k + 1, j] - v[k, j]) / l2) - g2x(v[k, j], (
                            v[k + 1, j] - v[k, j]) / l2, (v[k + 1, j] - 2 * v[k, j] + v[k - 1, j]) / l2 ** 2)) + (
                                      2 * v[k, j] - v[k, j - 1])
    u[1, j + 1] = u[2, j + 1] / 2
    v[n2 - 2, j + 1] = v[n2 - 3, j + 1] / 2
    f[1] = be2 * v[2, j + 1] / l2 ** 2 - u[n1 - 3, j + 1] / l1 ** 2
    f[2] = be2 * (v[3, j + 1] - 3 * v[2, j + 1]) / l2 ** 3 + (u[n1 - 4, j + 1] - 3 * u[n1 - 3, j + 1]) / l1 ** 3 + g1(
        u[i, j + 1], - (u[n1 - 3, j + 1] - u[n1 - 4, j + 1]) / l1) - g2(v[k, j + 1], - (v[3, j + 1] - v[2, j + 1]) / l2)
    st = np.linalg.solve(h, f)
    u[n1 - 2, j + 1] = st[0]
    u[n1 - 1, j + 1] = st[1]
    v[0, j + 1] = st[1]
    v[1, j + 1] = st[2]

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('t')
ax.set_ylabel('X')
ax.set_title('u(t,x)')
Xx, Yy = np.meshgrid(t, X)
z = np.vstack((u, v))
surf = ax.plot_surface(Xx, Yy, z, rstride=1, cstride=1, linewidth=0, cmap=mpl.cm.hsv)
fig.colorbar(surf, shrink=0.75, aspect=15)
plt.show()