import numpy as np
import drmpc
import control
import pickle
import scipy.linalg as linalg
import time

filename = "n2_radius"
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
trials = 30


def run_sim(Sigma_hat, epsilon):

    DRMPC = drmpc.DRMPC(
        params,
        N,
        epsilon=epsilon,
        Sigma_hat=Sigma_hat,
        iid=True,
        warmstart=True,
        K_f=K_f,
    )

    x0 = np.array([[1], [1]])
    Nsim = int(500)

    xcl = np.empty((trials, n, Nsim + 1))
    ucl = np.empty((trials, m, Nsim))
    wcl = np.empty((trials, q, Nsim))
    cost = np.empty((trials, Nsim))
    comp_times = np.empty((trials, Nsim))
    np.random.seed(123)

    Sigma = np.array([[0.01, 0.01], [0.01, 0.035]])
    for ii in range(trials):
        w_prev = None
        print(ii)
        xcl[ii, :, 0] = x0[:, 0]
        for k in range(Nsim):
            start = time.time()
            ucl[ii, :, k] = DRMPC.kappa(xcl[ii, :, k], w_prev=w_prev)
            end = time.time()
            comp_times[ii, k] = end - start
            wcl[ii, :, k] = (
                linalg.sqrtm(Sigma) @ (1 - 2 * np.random.rand(2)) * np.sqrt(3)
            )
            w_prev = wcl[ii, :, k]
            cost[ii, k] = (
                xcl[ii, :, k].T @ Q @ xcl[ii, :, k]
                + ucl[ii, :, k].T @ R @ ucl[ii, :, k]
            )

    sim = {
        "xcl": xcl,
        "ucl": ucl,
        "wcl": wcl,
        "cost": cost,
        "comp_times": comp_times,
    }
    return sim


Sigma_hat = np.diag([0.01, 0.01])
epsilons = np.logspace(-2, 0, 10)
J_i = np.empty((len(epsilons), trials))

for i in range(len(epsilons)):
    print(epsilons[i])
    drmpc_sim = run_sim(Sigma_hat, epsilons[i])
    J_i[i, :] = np.average(drmpc_sim["cost"], axis=-1)

stats = {"epsilons": epsilons, "J_i": J_i}

with open(filename + ".pkl", "wb") as handle:
    pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
