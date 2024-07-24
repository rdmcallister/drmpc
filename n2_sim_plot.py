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

drmpc = data["drmpc"]
smpc = data["smpc"]
rmpc = data["rmpc"]

Nsim = len(drmpc["ucl"][0, 0])

print(np.shape(drmpc["cost"]))


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
    K_f = np.zeros((m, n))

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
        iid=True,
        warmstart=True,
        K_f=K_f,
    )

    Sigma_max = DRMPC.bisection(G.T @ P @ G, rho, Sigma_hat)
    bound = np.trace(G.T @ P @ G @ Sigma_max)
    return bound


bound = calc_bound()

# ax = axs[0]
# ax.plot(np.average(drmpc["xcl"], axis=0).T, label="DRMPC")
# ax.plot(np.average(smpc["xcl"], axis=0).T, label="SMPC")
# ax.plot(np.average(rmpc["xcl"], axis=0).T, label="RMPC")
# ax.set_xlim(0, Nsim)
# ax.set_xlabel(r"$k$")
# ax.set_ylabel(r"$\hat{\mathbb{E}}[x_i(k)]$")


ax = axs[0]
drmpc_sq = np.power(np.linalg.norm(drmpc["xcl"], axis=1), 2)
smpc_sq = np.power(np.linalg.norm(smpc["xcl"], axis=1), 2)
rmpc_sq = np.power(np.linalg.norm(rmpc["xcl"], axis=1), 2)
ax.plot(
    np.average(drmpc_sq, axis=0),
    label="DRMPC",
    marker=".",
)
drmpc_sq_std = np.std(drmpc_sq, axis=0)
drmpc_sq_min = np.average(drmpc_sq, axis=0) - drmpc_sq_std
drmpc_sq_max = np.average(drmpc_sq, axis=0) + drmpc_sq_std
ax.fill_between(
    np.arange(len(drmpc_sq_min)),
    drmpc_sq_min,
    drmpc_sq_max,
    alpha=0.3,
)
ax.plot(
    np.average(smpc_sq, axis=0),
    label="SMPC",
    marker="*",
)
smpc_sq_std = np.std(smpc_sq, axis=0)
smpc_sq_min = np.average(smpc_sq, axis=0) - smpc_sq_std
smpc_sq_max = np.average(smpc_sq, axis=0) + smpc_sq_std
ax.fill_between(
    np.arange(len(smpc_sq_min)),
    smpc_sq_min,
    smpc_sq_max,
    alpha=0.3,
)
ax.plot(
    np.average(rmpc_sq, axis=0),
    label="RMPC",
    marker="x",
)
rmpc_sq_std = np.std(rmpc_sq, axis=0)
rmpc_sq_min = np.average(rmpc_sq, axis=0) - smpc_sq_std
rmpc_sq_max = np.average(rmpc_sq, axis=0) + smpc_sq_std
ax.fill_between(
    np.arange(len(rmpc_sq_min)),
    rmpc_sq_min,
    rmpc_sq_max,
    alpha=0.3,
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

drmpc_std = np.empty(Nsim + 1)
smpc_std = np.empty(Nsim + 1)
rmpc_std = np.empty(Nsim + 1)
for k in range(Nsim):
    drmpc_runcost[k] = np.average(drmpc["cost"][:, : (k + 1)])
    smpc_runcost[k] = np.average(smpc["cost"][:, : (k + 1)])
    rmpc_runcost[k] = np.average(rmpc["cost"][:, : (k + 1)])

    drmpc_std[k] = np.std(np.average(drmpc["cost"][:, : (k + 1)], axis=-1))
    smpc_std[k] = np.std(np.average(smpc["cost"][:, : (k + 1)], axis=-1))
    rmpc_std[k] = np.std(np.average(rmpc["cost"][:, : (k + 1)], axis=-1))

drmpc_runcost[Nsim] = drmpc_runcost[Nsim - 1]
smpc_runcost[Nsim] = smpc_runcost[Nsim - 1]
rmpc_runcost[Nsim] = rmpc_runcost[Nsim - 1]

drmpc_std[Nsim] = drmpc_std[Nsim - 1]
smpc_std[Nsim] = smpc_std[Nsim - 1]
rmpc_std[Nsim] = rmpc_std[Nsim - 1]

ax = axs[1]
ax.semilogy(drmpc_runcost, label="DRMPC", marker=".", markevery=10)
drmpc_min = drmpc_runcost - drmpc_std
drmpc_max = drmpc_runcost + drmpc_std
ax.fill_between(
    np.arange(len(drmpc_min)),
    drmpc_min,
    drmpc_max,
    alpha=0.3,
)
ax.semilogy(smpc_runcost, label="SMPC", marker="*", markevery=10)
smpc_min = smpc_runcost - smpc_std
smpc_max = smpc_runcost + smpc_std
ax.fill_between(
    np.arange(len(smpc_min)),
    smpc_min,
    smpc_max,
    alpha=0.3,
)
ax.semilogy(rmpc_runcost, label="RMPC", marker="x", markevery=10)
rmpc_min = rmpc_runcost - rmpc_std
rmpc_max = rmpc_runcost + rmpc_std
ax.fill_between(
    np.arange(len(rmpc_min)),
    rmpc_min,
    rmpc_max,
    alpha=0.3,
)
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

# ax = axs[1]
# ax.plot(np.average(drmpc["cost"], axis=0), label="DRMPC")
# ax.plot(np.average(smpc["cost"], axis=0), label="SMPC")
# ax.plot(np.average(rmpc["cost"], axis=0), label="RMPC")
# ax.set_xlim(0, len(drmpc["cost"][0]))
# ax.set_xlabel(r"$k$")
# ax.set_ylabel(r"$\ell(x,u)$")

# ax = axs[1]
# ax.plot(drmpc["cost"].T, label="DRMPC")
# ax.plot(smpc["cost"].T, label="SMPC")
# ax.plot(rmpc["cost"].T, label="RMPC")
# ax.set_xlim(0, len(drmpc["cost"][0]))
# ax.set_xlabel(r"$k$")
# ax.set_ylabel(r"$\ell(x,u)$")

fig.tight_layout(pad=0.5)

plt.show()
fig.savefig("n2_stats_ieee.pdf")
