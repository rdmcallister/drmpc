import matplotlib.pyplot as plt
import numpy as np
import pickle
import plottools
import control
import drmpc


with open("n2_sim.pkl", "rb") as handle:
    data = pickle.load(handle)

fig, axs = plt.subplots(2, 1, figsize=(3.5, 4))

gridstyle = {"linewidth": 0.5, "alpha": 0.5}

norot = {"rotation": 0, "va": "center", "ha": "right"}

drmpc_sim = data["drmpc"]
smpc_sim = data["smpc"]
rmpc_sim = data["rmpc"]

Nsim = len(drmpc_sim["ucl"][0, 0])

print(np.shape(drmpc_sim["cost"]))


def calc_bound():
    n = 2
    m = 2
    q = 2

    A = np.array([[0.9, 0], [0.2, 0.8]])
    B = np.array([[1, 0], [0, 1]])
    G = np.array([[1, 0], [0, 1]])

    C = np.zeros((4, 2))
    D = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    b = np.array([[1], [1], [0], [1]])

    C_f = np.zeros((1, 2))
    b_f = 0

    S = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    h = np.array([[1], [1], [1], [1]])

    Q = np.diag([0.1, 10])
    R = np.diag([10, 0.1])
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

    N = 10

    Sigma_hat = np.diag([0.01, 0.01])
    rho = 0.1

    DRMPC = drmpc.DRMPC(
        params,
        N,
        rho=rho,
        Sigma_hat=Sigma_hat,
    )

    Sigma_max = DRMPC.bisection(G.T @ P @ G, rho, Sigma_hat)
    bound = np.trace(G.T @ P @ G @ Sigma_max)
    return bound


bound = calc_bound()


ax = axs[0]
ax.plot(
    np.average(np.power(np.linalg.norm(drmpc_sim["xcl"], axis=1), 2), axis=0),
    label="DRMPC",
    marker=".",
)
ax.plot(
    np.average(np.power(np.linalg.norm(smpc_sim["xcl"], axis=1), 2), axis=0),
    label="SMPC",
    marker="*",
)
ax.plot(
    np.average(np.power(np.linalg.norm(rmpc_sim["xcl"], axis=1), 2), axis=0),
    label="RMPC",
    marker="x",
)
ax.set_xlim(0, 50)
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\tilde{\mathbb{E}}\left[|x(k)|^2\right]$", **norot)
ax.grid(**gridstyle)
ax.set_ylim(0, 1)
ax.legend()

drmpc_runcost = np.empty(Nsim + 1)
smpc_runcost = np.empty(Nsim + 1)
rmpc_runcost = np.empty(Nsim + 1)
for k in range(Nsim):
    drmpc_runcost[k] = np.average(drmpc_sim["cost"][:, : (k + 1)])
    smpc_runcost[k] = np.average(smpc_sim["cost"][:, : (k + 1)])
    rmpc_runcost[k] = np.average(rmpc_sim["cost"][:, : (k + 1)])

drmpc_runcost[Nsim] = drmpc_runcost[Nsim - 1]
smpc_runcost[Nsim] = smpc_runcost[Nsim - 1]
rmpc_runcost[Nsim] = rmpc_runcost[Nsim - 1]

ax = axs[1]
ax.semilogy(drmpc_runcost, label="DRMPC", marker=".", markevery=10)
ax.semilogy(smpc_runcost, label="SMPC", marker="*", markevery=10)
ax.semilogy(rmpc_runcost, label="RMPC", marker="x", markevery=10)
ax.axhline(
    bound,
    color="grey",
    linestyle="dashed",
    label=r"$\max_{\Sigma\in\mathbb{B}_d}\textrm{tr}(G'PG\Sigma)$",
)
ax.set_xlim(0, Nsim)
ax.set_xticks(np.arange(0, Nsim + 1, 100))
ax.set_ylim(0.5, 2)
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\tilde{\mathcal{J}}_k$", **norot)
ax.grid(**gridstyle, which="both")
ax.legend()

fig.tight_layout(pad=0.5)

plt.show()
fig.savefig("n2_stats.pdf")
