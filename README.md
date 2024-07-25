# DRMPC: Distributionally Robust Model Predictive Control
DRMPC contains algorithms for formulating and solving distributionally robust model predictive control (DRMPC) problems through the convex optimization package ``cvxpy``. The DRMPC problem considers a linear system with linear constraints and (convex) quadratic costs defined via the numpy arrays ``A, B, G, C, D, b, C_f, b_f, S, h``, a prediction horizon ``N``, and the ambiguity parameters ``epsilon, Sigma_hat``. The linear system is defined as 

$$x(k+1)=Ax(k)+Bu(k)+Gw(k)$$ 

in which $x(k)$ is the state, $u(k)$ is the input, and $w(k)$ is the disturbance at discrete time $k$. The linear constraints include a state/input constraint $Cx(k)+Bu(k)\leq b$, a polytopic support for the disturbance $Sw(k)\leq h$, and a linear terminal constraint $C_fx(N)\leq b_f$. Given this linear system and the initial state $x(0)$, the objective function is defined with the matrices $Q,R,P\succ 0$ as

$$
\Phi(x(0),\mathbf{u},\mathbf{w}) = x(N)'Px(N) + \sum_{k=0}^{N-1}x(k)'Qx(k) + u(k)'Ru(k)
$$

in which $\mathbf{u}=(u(0),\dots,u(N-1))$ and $\mathbf{w}=(w(0),\dots,w(N-1))$. The DRMPC problem considers the worst case distribution of $w(k)$ in an ambiguity set containing all zero-mean distributions within a Gelbrich ball centered at the covariance $\Sigma$ with radius $\varepsilon$ denoted $\mathcal{P}(\varepsilon,\widehat{\Sigma})$. 

The DRMPC problem is formulated as follows. We assume an affine disturbance feedback paramterization such that we optimize over the vectors $v(k)$ and matrices $M(k,j)$ that determine $u(k)$ as follows:

$$ u(k) = v(k) + \sum_{j=0}^{k-1}M(k,j)w(j)  \qquad \mathbf{u} = \mathbf{v} + \mathbf{M}\mathbf{w}$$

We denote the collection of $M(k,j)$ for all $k,j\in\{0,1,\dots,N-1\}$ as $\mathbf{M}$ and collection of $v(k)$ for all $k\in\{0,1,\dots,N-1\}$ as \mathbf{v}$. This parameterization with the linear system, linear constraints, and initial condition $x(0)$ leads to the constraint:

$$ (\mathbf{M},\mathbf{v})\in\Pi(x(0)) := \bigcap_{Sw(k)\leq h}\left\\{ (\mathbf{M},\mathbf{v}) \ \middle| \ \begin{matrix} Cx(k) + Du(k) \leq b \ \forall k \\\ C_fx(N)\leq b_f \end{matrix} \right\\} $$

The DRMPC optimization problem is therefore

$$
\min_{(\mathbf{M},\mathbf{v})\in\Pi(x(0))}\max_{\mathbb{P}(0),\dots,\mathbb{P}(N-1)\in \mathcal{P}(\varepsilon,\widehat{\Sigma})} \mathbb{E}\left[\Phi(x(0),\mathbf{v} + \mathbf{M}\mathbf{w},\mathbf{w}) \ \middle| \ w(k)\sim\mathbb{P}(k) \right] 
$$

DRMPC allows the user to easily formulate and solve this optimization problem to obtain the optimal solution $(\mathbf{M}^0(x),\mathbf{v}^0(x))$ for a given initial condition $x(0)=x$. In closed-loop implementation, the input to the system is defined as $v^0(0;x)$ in which $\mathbf{v}^0(x)=(v^0(0;x),\dots,v^0(N-1;x))$ to give the control law $\kappa(x) = v^0(0;x)$. Further details on the problem formulation can be found in the paper: [Distributionally Robust Model Predictive Control: Closed-loop Guarantees and Scalable Algorithms](https://arxiv.org/abs/2309.12758).

# Algorithms 

Two main algorithms are available in DRMPC to solve this optimization problem:

1) Semidefinite program (SDP) ``alg="SDP"``: This approach reformulates the min-max optimization problem via linear matrix inequalities such that it can be solved as an SDP using available solvers in ``cvxpy``. 
2) Newton-type algorithm (NT) ``alg="NT"``: This approach uses a new Newton-type algorithm tailored to this DRMPC optimization problem. The NT algorithm solves a quadratic program (QP) at each iteration using available QP solvers in ``cvxpy``. For larger problems, the NT algorithm has shown to be significantly faster than the SDP formulation. The NT algorithm is also an "anytime" algorithm in that each iterate is guaranteed to be a feasible solution such that the algorithm can be terminated, if needed, after only one iteration (i.e., one QP).

 Further details on the algorithms can be found in the paper: [Distributionally Robust Model Predictive Control: Closed-loop Guarantees and Scalable Algorithms](https://arxiv.org/abs/2309.12758).

# Requirements
DRMPC depends on the packages ``numpy``, ``cvxpy``, ``scipy``, and ``time``. For ``cvxpy``, we recommend installing the convex solver [MOSEK](https://www.mosek.com/) (free academic license) for best performance for the SDP formulation. However, standard solvers in ``cvxpy`` can also be used. For example, SCS can be used to solve the SDP formulation and OSQP can be used to solve the QPs in the NT algorithm. The solver used from ``cvxpy`` can be specified via the option ``solver=<solver name>`` when initializing the DRMPC object. To run the example scripts, the packages ``control``, ``matplotlib``, and ``pickle`` are also required. 

# Usage and examples
The python file ``drmpc.py`` contains the main class for the DRMPC controller (``DRMPC``) and all algorithms. To use these algorithms, simply move ``drmpc.py`` to your working directory and use ``import drmpc``. The example file ``n2_nom.py`` provides a good starting point to setup and test the DRMPC controller. The example scripts ``ex_convergence.py``, ``ex_comparison_nt_sdp.py``, ``ex_nom.py``, ``ex_sim.py``, and ``ex_radius.py`` correspond to Figures 1-5, respectively, in the paper: [Distributionally Robust Model Predictive Control: Closed-loop Guarantees and Scalable Algorithms](https://arxiv.org/abs/2309.12758). Note that ``ex_sim.py`` and ``ex_radius.py`` may take many hours to run given the number of simulations. 

# Citing
If you use DRMPC for research, please cite our accompanying paper:
```bibtex
@article{mcallister2023distributionally,
  title={Distributionally Robust Model Predictive Control: Closed-loop Guarantees and Scalable Algorithms},
  author={McAllister, Robert D and Esfahani, Peyman Mohajerin},
  journal={arXiv preprint arXiv:2309.12758},
  year={2023}
}
```
