import matplotlib.pyplot as plt
import numpy as np
import pickle


with open("n2_radius.pkl", "rb") as handle:
    data = pickle.load(handle)

fig, axs = plt.subplots(1, 1, figsize=(3.5, 2.5))

norot = {"rotation": 0, "va": "center", "ha": "right"}

J_i = data["J_i"]
rhos = data["rhos"]

J_avg = np.average(J_i, axis=-1)

ax = axs
ax.semilogx(
    rhos, np.average(J_i, axis=-1), label=r"$\tilde{\mathcal{J}}_{T}$", marker="."
)
J_min = np.average(J_i, axis=-1) - np.std(J_i, axis=-1)
J_max = np.average(J_i, axis=-1) + np.std(J_i, axis=-1)
ax.fill_between(rhos, J_min, J_max, alpha=0.3)

J_sorted = np.sort(J_i, axis=-1)
J_min = J_sorted[:, 0]
J_max = J_sorted[:, -1]
ax.semilogx(rhos, J_max, label=r"min/max", linestyle="dashed", color="C0")
ax.semilogx(rhos, J_min, linestyle="dashed", color="C0")
ax.set_xlim(rhos[0], rhos[-1])
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel(r"$\tilde{\mathcal{J}}_{T}$", **norot)
ax.grid(linewidth=0.5, alpha=0.5)
ax.legend()

fig.tight_layout(pad=0.5)

plt.show()
fig.savefig("n2_radius_plot.pdf")
