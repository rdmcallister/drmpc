import numpy as np
import drmpc_qpfw
import control
import pickle
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import time

filename = "n2_qpfw_nom"
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

N = 6


def run_sim(Sigma_hat, rho):

    DRMPC = drmpc_qpfw.DRMPC(
        params,
        N,
        rho=rho,
        Sigma_hat=Sigma_hat,
        iid=True,
        warmstart=True,
        K_f=K_f,
    )

    trials = 1
    x0 = np.array([[1], [1]])
    Nsim = int(50)

    xcl = np.empty((trials, n, Nsim + 1))
    ucl = np.empty((trials, m, Nsim))
    wcl = np.empty((trials, q, Nsim))
    cost = np.empty((trials, Nsim))
    comp_times = np.empty((trials, Nsim))
    np.random.seed(123)

    Sigma = np.array([[0.01, 0.01], [0.01, 0.04]])
    dist = "uniform"
    bias = False
    for ii in range(trials):
        w_prev = None
        print(ii)
        xcl[ii, :, 0] = x0[:, 0]
        for k in range(Nsim):
            start = time.time()
            ucl[ii, :, k] = DRMPC.kappa(xcl[ii, :, k], w_prev=w_prev)
            end = time.time()
            comp_times[ii, k] = end - start
            wcl[ii, :, k] = np.zeros(q)
            w_prev = wcl[ii, :, k]
            xcl[ii, :, k + 1] = (
                A @ xcl[ii, :, k] + B @ ucl[ii, :, k] + G @ wcl[ii, :, k]
            )
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
rho = 0.1

drmpc_sim = run_sim(Sigma_hat, rho)
smpc_sim = run_sim(Sigma_hat, 0)
rmpc_sim = run_sim(np.zeros((2, 2)), 0)

stats = {"drmpc": drmpc_sim, "smpc": smpc_sim, "rmpc": rmpc_sim}

with open(filename + ".pkl", "wb") as handle:
    pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
