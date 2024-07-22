import numpy as np
import drmpc
import control
import pickle
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import time

n = 2  # state dimension
m = 2  # input dimension
q = 2  # disturbance dimension

# x^+ = Ax + Bu + Gw
A = np.array([[0.9, 0], [0.2, 0.8]])
B = np.array([[1, 0], [0, 1]])
G = np.array([[1, 0], [0, 1]])

# Cx(k) + Du(k) <= b
C = np.zeros((4, 2))
# D = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
# b = np.array([[1], [1], [0], [1]])
D = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
b = np.array([[1], [1], [0], [1]])

# x(N)C_f <= b_f
C_f = np.zeros((1, 2))
b_f = 0

# Sw <= h
S = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
h = np.array([[1], [1], [1], [1]])

# \ell(x,u) = x'Qx + u'Ru
Q = np.diag([0.1, 10])
R = np.diag([10, 0.1])
# V_f(x) = x'Px
P = control.dlyap(A, Q)


params = {
    "A": A,
    "B": B,
    "G": G,
    "C": C,
    "D": D,
    "b": b,
    "C_f": C_f,
    "b_f": b_f,
    "S": S,
    "h": h,
    "Q": Q,
    "R": R,
    "P": P,
}

# Horizon length
N = 10

Sigma_hat = np.diag([0.01, 0.01])
rho = 0.1

x0 = np.array([[1], [1]])
v_ws = np.zeros((N * m, 1))
M_ws = np.zeros((N * m, N * q))

## SDP baseline
DRMPC = drmpc.DRMPC(
    params,
    N,
    rho=rho,
    Sigma_hat=Sigma_hat,
    warmstart=False,
    alg="SDP",
)

output = DRMPC.solve_ocp(
    x0,
    v_ws=v_ws,
    M_ws=M_ws,
)
obj = output["opt"]["obj"]


def run_opt(alg, stepsize):
    DRMPC = drmpc.DRMPC(
        params,
        N,
        rho=rho,
        Sigma_hat=Sigma_hat,
        warmstart=False,
        alg=alg,
        alg_options={"max_iter": 100, "stepsize": stepsize, "tol": 1e-8},
    )

    output = DRMPC.solve_ocp(
        x0,
        v_ws=v_ws,
        M_ws=M_ws,
    )
    return output["opt"]["cost"]


fw_a = run_opt("FW", "adaptive")
fw_fa = run_opt("FW", "fully adaptive")
nt_a = run_opt("NT", "adaptive")
nt_fa = run_opt("NT", "fully adaptive")

gridstyle = {"linewidth": 0.5, "alpha": 0.5, "which": "major"}


fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), sharey=True)

ax.semilogy(
    np.arange(len(fw_a)),
    fw_a - obj,
    label="FW Adaptive",
    marker=".",
)
ax.semilogy(
    np.arange(len(fw_fa)),
    fw_fa - obj,
    label="FW Fully Adaptive",
    marker="d",
)

ax.semilogy(
    np.arange(len(nt_a)),
    nt_a - obj,
    label="NT Adaptive",
    marker="x",
)
ax.semilogy(
    np.arange(len(nt_fa)),
    nt_fa - obj,
    label="NT Fully Adaptive",
    marker="*",
)

ax.set_xlabel("iteration")
ax.set_ylabel(r"$f(\theta_t)-f^*$")
ax.set_xlim(0, 100)
ax.set_ylim(1e-6, 50)
ax.legend()
ax.grid(**gridstyle)
plt.show()
