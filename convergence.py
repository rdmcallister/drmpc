import numpy as np
import drmpc
import control
import matplotlib.pyplot as plt

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

output = DRMPC.solve_ocp(x0)
obj = output["opt"]["obj"]


def run_opt(alg, stepsize, max_iter):
    DRMPC = drmpc.DRMPC(
        params,
        N,
        rho=rho,
        Sigma_hat=Sigma_hat,
        warmstart=False,
        alg=alg,
        alg_options={"max_iter": max_iter, "stepsize": stepsize, "tol": 1e-8},
    )

    output = DRMPC.solve_ocp(
        x0,
        v_ws=v_ws,
        M_ws=M_ws,
    )
    return output["opt"]


fw_a = run_opt("FW", "adaptive", 1000)
fw_fa = run_opt("FW", "fully adaptive", 100)
nt_a = run_opt("NT", "adaptive", 100)
nt_fa = run_opt("NT", "fully adaptive", 100)


gridstyle = {"linewidth": 0.5, "alpha": 0.5, "which": "major"}


fig, axs = plt.subplots(2, 1, figsize=(3.5, 4), sharey=True)

ax = axs[0]
ax.semilogy(
    np.arange(len(fw_a["cost"])),
    fw_a["cost"] - obj,
    label="FW-A",
    marker=".",
)
ax.semilogy(
    np.arange(len(fw_fa["cost"])),
    fw_fa["cost"] - obj,
    label="FW-FA",
    marker="d",
)

ax.semilogy(
    np.arange(len(nt_a["cost"])),
    nt_a["cost"] - obj,
    label="NT-A",
    marker="x",
)
ax.semilogy(
    np.arange(len(nt_fa["cost"])),
    nt_fa["cost"] - obj,
    label="NT-FA",
    marker="*",
)

ax.set_xlabel("iteration")
ax.set_ylabel(r"$f(\theta_t)-f^*$")
ax.set_xlim(0, 100)
ax.set_ylim(1e-6, 50)
ax.legend()
ax.grid(**gridstyle)

ax = axs[1]
ax.semilogy(
    fw_a["iter_time"],
    fw_a["cost"] - obj,
    label="FW-A",
    marker=".",
)
ax.semilogy(
    fw_fa["iter_time"],
    fw_fa["cost"] - obj,
    label="FW-FA",
    marker="d",
)

ax.semilogy(
    nt_a["iter_time"],
    nt_a["cost"] - obj,
    label="NT-A",
    marker="x",
)
ax.semilogy(
    nt_fa["iter_time"],
    nt_fa["cost"] - obj,
    label="NT-FA",
    marker="*",
)

ax.set_xlabel("computation time (s)")
ax.set_ylabel(r"$f(\theta_t)-f^*$")
ax.set_xlim(0, 10)
ax.set_ylim(1e-6, 50)
ax.grid(**gridstyle)

fig.tight_layout(pad=0.5)

plt.show()

fig.savefig("convergence_plot.pdf")
