import matplotlib.pyplot as plt
import numpy as np
import pickle


with open("comparison.pkl", "rb") as handle:
    data = pickle.load(handle)

# Check if the optimal cost for NT is larger than SDP by more than 1e-6 (tolerance)
print(np.array(data["obj_nt"]) - np.array(data["obj_sdp"]) > 1e-6)

Ns = data["Ns"]
time_sdp = data["time_sdp"]
time_nt = data["time_nt"]

fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))

ax.plot(Ns, time_sdp, label="MOSEK (LMI)", marker="s", color="k")
ax.plot(Ns, time_nt, label="NT-FA", marker="*", color="C3")
ax.set_xlabel(r"$N$")
ax.set_ylabel("computation time (s)")
ax.set_xlim(Ns[0], Ns[-1])
ax.legend()
ax.grid(alpha=0.5, linewidth=0.5)
ax.set_ylim(0, np.max(time_sdp) * 1.05)
fig.tight_layout()

plt.show()

fig.savefig("comparison_plot.pdf")
