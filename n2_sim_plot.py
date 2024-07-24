import matplotlib.pyplot as plt
import numpy as np
import pickle
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

ax = axs[0]
drmpc_sq = np.power(np.linalg.norm(drmpc_sim["xcl"], axis=1), 2)
smpc_sq = np.power(np.linalg.norm(smpc_sim["xcl"], axis=1), 2)
rmpc_sq = np.power(np.linalg.norm(rmpc_sim["xcl"], axis=1), 2)
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
    drmpc_runcost[k] = np.average(drmpc_sim["cost"][:, : (k + 1)])
    smpc_runcost[k] = np.average(smpc_sim["cost"][:, : (k + 1)])
    rmpc_runcost[k] = np.average(rmpc_sim["cost"][:, : (k + 1)])

    drmpc_std[k] = np.std(np.average(drmpc_sim["cost"][:, : (k + 1)], axis=-1))
    smpc_std[k] = np.std(np.average(smpc_sim["cost"][:, : (k + 1)], axis=-1))
    rmpc_std[k] = np.std(np.average(rmpc_sim["cost"][:, : (k + 1)], axis=-1))

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

ax.set_xlim(0, Nsim)
ax.set_xticks(np.arange(0, Nsim + 1, 100))
ax.set_ylim(0.5, 2)
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\tilde{\mathcal{J}}_k$", **norot)
ax.grid(**gridstyle, which="both")
ax.legend()

fig.tight_layout()

plt.show()
fig.savefig("n2_stats_ieee.pdf")
