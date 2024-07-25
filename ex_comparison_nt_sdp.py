import numpy as np
import control
import drmpc
import matplotlib.pyplot as plt
import time
import pickle

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

Sigma_hat = np.diag([0.01, 0.01])
epsilon = 0.1
max_iter = 1000
tol = 1e-6
N_max = 30

x0 = np.array([[1], [1]])

time_sdp = []
time_nt = []
obj_sdp = []
obj_nt = []

for N in range(1, N_max + 1):
    print(N)

    ## SDP baseline
    DRMPC_SDP = drmpc.DRMPC(
        params,
        N,
        epsilon=epsilon,
        Sigma_hat=Sigma_hat,
        warmstart=False,
        alg="SDP",
    )

    start = time.time()
    output = DRMPC_SDP.solve_ocp(x0)
    time_sdp.append(time.time() - start)
    obj_sdp.append(output["opt"]["obj"])

    ## NT
    DRMPC_NT = drmpc.DRMPC(
        params,
        N,
        epsilon=epsilon,
        Sigma_hat=Sigma_hat,
        warmstart=False,
        alg="NT",
        alg_options={"max_iter": max_iter, "tol": tol},
    )

    start = time.time()
    output = DRMPC_NT.solve_ocp(x0)
    time_nt.append(time.time() - start)
    obj_nt.append(output["opt"]["obj"])


Ns = np.arange(1, N_max + 1)

stats = {
    "Ns": Ns,
    "time_sdp": time_sdp,
    "time_nt": time_nt,
    "obj_sdp": obj_sdp,
    "obj_nt": obj_nt,
}

with open("comparison.pkl", "wb") as handle:
    pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
