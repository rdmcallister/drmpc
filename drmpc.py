import numpy as np
import scipy as sp
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
import scipy.linalg as linalg
import time


class DRMPC:
    def __init__(
        self,
        params,
        N,
        rho=None,
        Sigma_hat=None,
        iid=False,
        bfSigma_hat=None,
        bfrho=None,
        warmstart=False,
        K_f=None,
    ):
        self.params = params
        self.N = N
        self.warmstart = warmstart
        self.M_prev = None
        self.v_prev = None
        self.K_f = K_f
        self.dims = self._get_dims(**params)
        self.matrices = self._construct_matrices(**{**params, **self.dims, "N": N})

        self.qp = self._construct_qp(**self.matrices)
        # self.lp = self._construct_feas_lp(**self.matrices)

        if rho is None and bfrho is None:
            rho = 0
        if Sigma_hat is None and bfSigma_hat is None:
            Sigma_hat = np.zeros((self.dims["q"], self.dims["q"]))

        if bfSigma_hat is None:
            bfSigma_hat = np.kron(np.eye(N), Sigma_hat)
        if bfrho is None:
            bfrho = np.sqrt(N) * rho

        self.rho = rho
        self.Sigma_hat = Sigma_hat
        self.bfrho = bfrho
        self.bfSigma_hat = bfSigma_hat

        self.iid = iid

        if iid:
            self.beta = self._estimate_beta_iid(
                **self.matrices, Sigma_hat=Sigma_hat, rho=rho
            )
        else:
            self.beta = self._estimate_beta(
                **self.matrices, bfSigma_hat=bfSigma_hat, bfrho=bfrho
            )

    def get_matrices(self):
        return self.matrices

    def _get_dims(self, B, G, **kwargs):
        (n, m) = np.shape(B)
        q = np.shape(G)[-1]
        return {"n": n, "m": m, "q": q}

    def _construct_matrices(
        self,
        A,
        B,
        G,
        C,
        D,
        b,
        C_f,
        b_f,
        S,
        h,
        Q,
        R,
        P,
        n,
        m,
        q,
        N,
    ):
        bfA = np.eye(n)
        bfL = np.zeros((n, n * N))
        for i in range(N):
            bfA = np.vstack([bfA, np.linalg.matrix_power(A, i + 1)])

        bfL = np.vstack([np.zeros((n, n)), bfA[: n * (N), :]])
        for j in range(N - 1):
            bfL_j = np.vstack([np.zeros((n * (j + 2), n)), bfA[: n * (N - j - 1), :]])
            bfL = np.hstack([bfL, bfL_j])
        bfB = np.matmul(bfL, np.kron(np.eye(N), B))
        bfG = np.matmul(bfL, np.kron(np.eye(N), G))

        (C_rows, C_cols) = np.shape(C)
        (Cf_rows, Cf_cols) = np.shape(C_f)
        bfC = np.block(
            [
                [np.kron(np.eye(N), C), np.zeros((N * C_rows, Cf_cols))],
                [np.zeros((Cf_rows, N * C_cols)), C_f],
            ]
        )

        (D_rows, D_cols) = np.shape(D)
        bfD = np.vstack([np.kron(np.eye(N), D), np.zeros((Cf_rows, N * D_cols))])
        c = np.vstack([np.kron(np.ones((N, 1)), b), b_f])

        F = np.matmul(bfC, bfB) + bfD
        E = np.matmul(bfC, bfG)
        H = -np.matmul(bfC, bfA)

        bfS = np.kron(np.eye(N), S)
        bfh = np.kron(np.ones((N, 1)), h)

        Q_D, Q_V = np.linalg.eigh(Q)
        eps = 1e-8
        Q_D[Q_D < eps] = 0
        Q_sqrt = (Q_V * np.sqrt(Q_D)) @ Q_V.T

        bfQ_sqrt = np.block(
            [
                [np.kron(np.eye(N), Q_sqrt), np.zeros((n * N, n))],
                [np.zeros((n, n * N)), linalg.cholesky(P)],
            ]
        )
        bfR_sqrt = np.kron(np.eye(N), linalg.cholesky(R))

        H_x = np.vstack([np.matmul(bfQ_sqrt, bfA), np.zeros((m * N, n))])
        H_u = np.vstack([np.matmul(bfQ_sqrt, bfB), bfR_sqrt])
        H_w = np.vstack([np.matmul(bfQ_sqrt, bfG), np.zeros((m * N, q * N))])

        matrices = {
            "F": F,
            "E": E,
            "H": H,
            "c": c,
            "bfS": bfS,
            "bfh": bfh,
            "H_x": H_x,
            "H_u": H_u,
            "H_w": H_w,
            "N": N,
            "n": n,
            "m": m,
            "q": q,
        }
        return matrices

    def solve_ocp(self, x0, w_prev=None):
        x0 = np.reshape(x0, (self.dims["n"], 1))
        if w_prev is not None:
            M_ws, v_ws = self.get_warmstart(x0, w_prev, self.M_prev, self.v_prev)
            # print(self._check_feas(self.lp, x0, M_ws, v_ws))
        else:
            M_ws = None
            v_ws = None
            if self.warmstart:
                print("Not using warmstart because w_prev was not provided.")
        start = time.time()
        opt = self.qpfw(
            x0,
            M_ws=M_ws,
            v_ws=v_ws,
            iter=100,
            tol=1e-8,
            stepsize="fully adaptive",
            tau=1.1,
            zeta=10,
        )
        end = time.time()
        dur = end - start
        kappa = opt["v"][0 : self.dims["m"], 0]
        if self.warmstart:
            self.M_prev = opt["M"]
            self.v_prev = opt["v"]
            # print(self._check_feas(self.lp, x0, self.M_prev, self.v_prev))
        return {"kappa": kappa, "opt": opt, "dur": dur}

    def kappa(self, x0, w_prev=None):
        output = self.solve_ocp(x0, w_prev=w_prev)
        return output["kappa"]

    def _construct_feas_lp(
        self,
        F,
        E,
        H,
        c,
        bfS,
        bfh,
        H_x,
        H_u,
        H_w,
        N,
        n,
        m,
        q,
        **kwargs,
    ):

        (S_rows, S_cols) = np.shape(bfS)
        (F_rows, F_cols) = np.shape(F)

        v = cp.Parameter((m * N, 1), name="v")
        M = cp.Parameter((N * m, q * N), name="M")
        bfZ = cp.Variable((S_rows, F_rows), name="bfZ")
        s = cp.Variable((F_rows, 1), name="s")

        x = cp.Parameter((n, 1), name="x")

        constraints = []
        constraints += [bfZ >= 0]
        constraints += [s >= 0]
        for i in range(N):
            constraints += [M[(m * i) : (m * (i + 1)), (q * i) :] == 0]

        constraints += [F @ v + bfZ.T @ bfh <= c + H @ x + s]
        constraints += [bfZ.T @ bfS == F @ M + E]
        obj = np.ones((1, F_rows)) @ s

        prob = cp.Problem(cp.Minimize(obj), constraints)
        lp = {"prob": prob, "v": v, "M": M, "x": x}
        return lp

    def _check_feas(self, lp, x, M, v):
        prob = lp["prob"]
        v_par = lp["v"]
        M_par = lp["M"]
        x_par = lp["x"]

        x_par.value = np.reshape(x, (len(x), 1))
        v_par.value = np.reshape(v, (len(v), 1))
        M_par.value = M

        prob.solve("MOSEK", warm_start=True)

        if prob.status not in ["infeasible", "unbounded"]:
            return prob.value
        else:
            raise ValueError("Problem is " + str(prob.status))

    def _construct_qp(
        self,
        F,
        E,
        H,
        c,
        bfS,
        bfh,
        H_x,
        H_u,
        H_w,
        N,
        n,
        m,
        q,
        **kwargs,
    ):

        (S_rows, S_cols) = np.shape(bfS)
        (F_rows, F_cols) = np.shape(F)
        (H_u_rows, H_u_cols) = np.shape(H_u)

        v = cp.Variable((m * N, 1), name="v")
        M = cp.Variable((N * m, q * N), name="M")
        bfZ = cp.Variable((S_rows, F_rows), name="bfZ")

        x = cp.Parameter((n, 1), name="x")
        bfSigma_sqrt = cp.Parameter((q * N, q * N), name="bfSigma_sqrt")

        constraints = []
        constraints += [bfZ >= 0]
        for i in range(N):
            constraints += [M[(m * i) : (m * (i + 1)), (q * i) :] == 0]

        constraints += [F @ v + bfZ.T @ bfh <= c + H @ x]
        constraints += [bfZ.T @ bfS == F @ M + E]
        obj = 2 * (H_x @ x).T @ H_u @ v + cp.quad_form(v, H_u.T @ H_u)

        Y = (H_u @ M + H_w) @ bfSigma_sqrt
        y = cp.vec(Y)
        obj += cp.quad_form(y, psd_wrap(np.eye(np.shape(y)[0])))

        prob = cp.Problem(cp.Minimize(obj), constraints)
        qp = {"prob": prob, "v": v, "M": M, "x": x, "bfSigma_sqrt": bfSigma_sqrt}
        return qp

    def _solve_qp(self, qp, x, bfSigma):
        prob = qp["prob"]
        v = qp["v"]
        M = qp["M"]
        x_par = qp["x"]
        bfSigma_sqrt_par = qp["bfSigma_sqrt"]

        x_par.value = np.reshape(x, (len(x), 1))
        bfSigma_sqrt_par.value = linalg.sqrtm(bfSigma)

        prob.solve("MOSEK", warm_start=True)

        if prob.status not in ["infeasible", "unbounded"]:
            return v.value, M.value
        else:
            raise ValueError("Problem is " + str(prob.status))

    def _estimate_beta_iid(self, H_u, Sigma_hat, rho, **kwargs):

        D = H_u.T @ H_u
        beta_v = 2 * np.linalg.norm(D, ord=2)
        beta_M = (np.linalg.norm(Sigma_hat, ord=2) + rho**2) * beta_v
        beta = max(beta_v, beta_M)
        return beta

    def _estimate_beta(self, H_u, bfSigma_hat, bfrho, **kwargs):
        D = H_u.T @ H_u
        beta_v = 2 * np.linalg.norm(D, ord=2)
        beta_M = (np.linalg.norm(bfSigma_hat, ord=2) + bfrho**2) * beta_v
        beta = max(beta_v, beta_M)
        return beta

    def bisection_iid(self, D, rho, Sigma_hat, N, tol=1e-8, iter=int(1e6)):
        q = np.shape(Sigma_hat)[0]
        bfSigma = np.zeros((q * N, q * N))
        for k in range(N):
            bfSigma[k * q : (k + 1) * q, k * q : (k + 1) * q] = self.bisection(
                D[k * q : (k + 1) * q, k * q : (k + 1) * q],
                rho,
                Sigma_hat,
                tol=tol,
                iter=iter,
            )
        return bfSigma

    def bisection(self, D, rho, Sigma_hat, tol=1e-8, iter=int(1e6)):
        n = np.shape(D)[0]
        [Lambda, V] = np.linalg.eigh(D)
        idx = np.argsort(Lambda)
        Lambda = Lambda[idx]

        # if D=0, all solutions are optimal, choose Sigma_hat
        if (np.abs(Lambda) <= 1e-8).all():
            return Sigma_hat
        # if rho is sufficiently small, ignore bisection step
        if rho <= 1e-16:
            return Sigma_hat

        V = V[:, idx]
        lambda_max = Lambda[-1]
        v_max = V[:, -1]
        LB = lambda_max * (1 + np.sqrt(v_max.T @ Sigma_hat @ v_max) / rho)
        UB = lambda_max * (1 + np.sqrt(np.trace(Sigma_hat)) / rho)
        for iter in range(iter):
            gamma = (LB + UB) / 2
            inv_gamma_minus_D = (V * (1 / (gamma - Lambda))) @ V.T
            L = gamma**2 * (inv_gamma_minus_D @ Sigma_hat @ inv_gamma_minus_D)
            phi = rho**2 - np.trace(
                Sigma_hat
                @ np.linalg.matrix_power(np.eye(n) - gamma * inv_gamma_minus_D, 2)
            )
            if phi < 0:
                LB = gamma
            else:
                UB = gamma
            Delta = (
                gamma * (rho**2 - np.trace(Sigma_hat))
                - np.trace(L @ D)
                + gamma**2 * np.trace(inv_gamma_minus_D @ Sigma_hat)
            )
            if phi > 0 and Delta < tol:
                break
        return L

    def get_warmstart(self, x_plus, w, M, v):

        A = self.params["A"]
        B = self.params["B"]
        G = self.params["G"]
        n = self.dims["n"]
        m = self.dims["m"]
        q = self.dims["q"]
        N = self.N
        K_f = self.K_f

        if K_f is None:
            raise ValueError("K_f must be specified to use warmstart method")

        x_plus = np.reshape(x_plus, (n, 1))
        w = np.reshape(w, (q, 1))
        M_ws = np.zeros_like(M)
        v_ws = np.zeros_like(v)
        for i in range(2, N):
            for j in range(1, i):
                M_ws[(i - 1) * m : (i) * m, (j - 1) * q : (j) * q] = M[
                    (i) * m : (i + 1) * m, (j) * q : (j + 1) * q
                ]

        sumv = np.zeros((n, 1))
        for i in range(1, N):
            v_ws[(i - 1) * m : (i) * m] = (
                v[(i) * m : (i + 1) * m] + M[(i) * m : (i + 1) * m, 0:q] @ w
            )
            sumv += (
                np.linalg.matrix_power(A, N - 1 - i) @ B @ v_ws[(i - 1) * m : (i) * m]
            )
        v_ws[(N - 1) * m : (N) * m] = K_f @ (
            np.linalg.matrix_power(A, N - 1) @ x_plus + sumv
        )

        for i in range(1, N):
            sumM = np.zeros((m, q))
            for j in range(i + 1, N):
                sumM += (
                    np.linalg.matrix_power(A, N - 1 - j)
                    @ B
                    @ M[j * m : (j + 1) * m, i * q : (i + 1) * q]
                )
            M_ws[(N - 1) * m : (N) * m, (i - 1) * q : (i) * q] = K_f @ (
                sumM + np.linalg.matrix_power(A, N - 1 - i) @ G
            )

        return M_ws, v_ws

    def calc_cost(self, x, v, M):
        H_x = self.matrices["H_x"]
        H_u = self.matrices["H_u"]
        H_w = self.matrices["H_w"]
        D = (H_u @ M + H_w).T @ (H_u @ M + H_w)
        if self.iid:
            bfSigma = self.bisection_iid(D, self.rho, self.Sigma_hat, self.N)
        else:
            bfSigma = self.bisection(D, self.bfrho, self.bfSigma_hat)
        out = 2 * x.T @ H_x.T @ H_u @ v + v.T @ H_u.T @ H_u @ v
        out += np.trace((H_u @ M + H_w).T @ (H_u @ M + H_w) @ bfSigma)
        return out.item()

    def qpfw(
        self,
        x,
        M_ws=None,
        v_ws=None,
        iter=100,
        tol=1e-8,
        stepsize="standard",
        tau=1.1,
        zeta=10,
    ):
        n = self.dims["n"]
        m = self.dims["m"]
        q = self.dims["q"]
        N = self.N
        beta = self.beta

        H_x = self.matrices["H_x"]
        H_u = self.matrices["H_u"]
        H_w = self.matrices["H_w"]

        iid = self.iid

        x = np.reshape(x, (n, 1))

        if M_ws is None or v_ws is None:
            [v, M] = self._solve_qp(self.qp, x, self.bfSigma_hat)
        else:
            M = M_ws
            v = np.reshape(v_ws, (N * m, 1))

        cost_ls = []
        dual_ls = []
        surrogate_dual_gap_ls = []
        dual_t = -np.inf

        def cost(v, M, bfSigma):
            out = 2 * x.T @ H_x.T @ H_u @ v + v.T @ H_u.T @ H_u @ v
            out += np.trace((H_u @ M + H_w).T @ (H_u @ M + H_w) @ bfSigma)
            return out.item()

        for t in range(iter):
            # solve inner maximization
            D = (H_u @ M + H_w).T @ (H_u @ M + H_w)
            if iid:
                bfSigma = self.bisection_iid(D, self.rho, self.Sigma_hat, N)
            else:
                bfSigma = self.bisection(D, self.bfrho, self.bfSigma_hat)

            cost_t = cost(v, M, bfSigma)
            cost_ls += [cost_t]

            # solve outer minimization
            [s_v, s_M] = self._solve_qp(self.qp, x, bfSigma)

            cost_qp = cost(s_v, s_M, bfSigma)

            # Evaluate dual gap
            deltaV_v = 2 * H_u.T @ H_x @ x + 2 * H_u.T @ H_u @ v
            deltaV_M = 2 * (H_u.T @ H_u) @ M @ bfSigma + 2 * H_u.T @ H_w @ bfSigma
            g = ((v - s_v).T @ deltaV_v + np.trace((M - s_M).T @ deltaV_M)).item()
            sq_norm_d = (
                (v - s_v).T @ (v - s_v) + np.trace((M - s_M).T @ (M - s_M))
            ).item()

            deltaV_v = 2 * H_u.T @ H_x @ x + 2 * H_u.T @ H_u @ v
            deltaV_M = 2 * (H_u.T @ H_u) @ M @ bfSigma + 2 * H_u.T @ H_w @ bfSigma
            g = ((v - s_v).T @ deltaV_v + np.trace((M - s_M).T @ deltaV_M)).item()
            sq_norm_d = (
                (v - s_v).T @ (v - s_v) + np.trace((M - s_M).T @ (M - s_M))
            ).item()

            dual_t = max(dual_t, cost_qp)
            dual_ls += [dual_t]
            surrogate_dual_gap_ls += [g]
            if cost_t - dual_t < tol:
                break

            if self.rho == 0:
                v = s_v
                M = s_M
                break

            if stepsize == "constant":
                eta = 1e-1
            elif stepsize == "standard":
                eta = 2 / (2 + t)
            elif stepsize == "adaptive":
                eta = min(1, g / beta / sq_norm_d)
            elif stepsize == "fully adaptive":
                beta = beta / zeta
                eta = min(1, g / beta / sq_norm_d)
                v_plus = (1 - eta) * v + eta * s_v
                M_plus = (1 - eta) * M + eta * s_M
                D_plus = (H_u @ M_plus + H_w).T @ (H_u @ M_plus + H_w)
                if iid:
                    bfSigma_plus = self.bisection_iid(
                        D_plus, self.rho, self.Sigma_hat, N
                    )
                else:
                    bfSigma_plus = self.bisection(D_plus, self.bfrho, self.bfSigma_hat)
                while (
                    cost(v_plus, M_plus, bfSigma_plus)
                    > cost_t - eta * g + eta**2 * beta / 2 * sq_norm_d
                ):
                    beta = tau * beta
                    eta = min(1, g / beta / sq_norm_d)
                    v_plus = (1 - eta) * v + eta * s_v
                    M_plus = (1 - eta) * M + eta * s_M
                    D_plus = (H_u @ M_plus + H_w).T @ (H_u @ M_plus + H_w)
                    if iid:
                        bfSigma_plus = self.bisection_iid(
                            D_plus, self.rho, self.Sigma_hat, N
                        )
                    else:
                        bfSigma_plus = self.bisection(
                            D_plus, self.bfrho, self.bfSigma_hat
                        )

            else:
                raise ValueError(
                    "Must specify valid stepsize type: constant, standard, adaptive, fully adaptive"
                )
            v = (1 - eta) * v + eta * s_v
            M = (1 - eta) * M + eta * s_M

        return {
            "v": v,
            "M": M,
            "cost": np.array(cost_ls),
            "dual": np.array(dual_ls),
            "surrogate_dual_gap": np.array(surrogate_dual_gap_ls),
        }
