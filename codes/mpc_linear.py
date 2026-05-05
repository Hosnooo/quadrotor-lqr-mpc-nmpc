#!/usr/bin/env python3
from __future__ import annotations
"""
mpc_linear.py
Linear MPC for the quadrotor hover LTI model (deviation form).

- CVXPY + OSQP backend
- ZOH discretization should use utils.c2d_series from this repo
- Costs in deviation variables: x̃ = x - x_ref, ũ = u - u_ref
- Supports box constraints on u and optional |Δu| ≤ du_max
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are

# local repo helpers (types only; actual discretization done outside)
from utils import c2d_series

@dataclass
class LinearMPCConfig:
    Ad: np.ndarray
    Bd: np.ndarray
    Q:  np.ndarray
    R:  np.ndarray
    P:  np.ndarray
    N:  int
    umin: np.ndarray
    umax: np.ndarray
    xmin: Optional[np.ndarray] = None
    xmax: Optional[np.ndarray] = None
    du_max: Optional[np.ndarray] = None   # elementwise |Δu| bound

class LinearMPC:
    """Warm-started linear MPC in deviation form."""
    def __init__(self, cfg: LinearMPCConfig):
        self.cfg = cfg
        Ad, Bd, N = cfg.Ad, cfg.Bd, cfg.N
        n, m = Bd.shape

        # Decision variables
        X = cp.Variable((n, N+1))
        U = cp.Variable((m, N))

        # Parameters (per-solve)
        x0 = cp.Parameter(n)  # measured x(0) (absolute)
        xr = cp.Parameter(n)  # reference x
        ur = cp.Parameter(m)  # reference u
        u_prev = cp.Parameter(m) if cfg.du_max is not None else None

        constr = [X[:, 0] == x0]
        cost = 0

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            # dynamics
            constr += [X[:, k+1] == Ad @ xk + Bd @ uk]

            # input box (absolute via deviation: u = ur + ũ)
            constr += [cfg.umin - ur <= uk, uk <= cfg.umax - ur]

            # state box (absolute via deviation: x = xr + x̃)
            if cfg.xmin is not None and cfg.xmax is not None:
                constr += [cfg.xmin - xr <= xk, xk <= cfg.xmax - xr]

            # slew rate
            if cfg.du_max is not None:
                if k == 0:
                    constr += [cp.abs(uk - (u_prev - ur)) <= cfg.du_max]
                else:
                    constr += [cp.abs(uk - U[:, k-1]) <= cfg.du_max]

            # stage cost
            cost += cp.quad_form(xk, cfg.Q) + cp.quad_form(uk, cfg.R)

        # terminal cost
        cost += cp.quad_form(X[:, N], cfg.P)

        self.X, self.U = X, U
        self.x0, self.xr, self.ur, self.u_prev = x0, xr, ur, u_prev
        self.problem = cp.Problem(cp.Minimize(cost), constr)
        self.solve_kwargs = dict(solver=cp.OSQP, warm_start=True, verbose=False)

    def make_step(self, x_meas: np.ndarray, xr: np.ndarray, ur: np.ndarray, u_prev: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve the QP and return the **absolute** control action u0 = ur + ũ0.
        Inputs:
          - x_meas: absolute x(0)
          - xr, ur: absolute references
          - u_prev: absolute u(-1) (only if du_max is set)
        """
        self.x0.value = np.asarray(x_meas, float) - np.asarray(xr, float)  # use deviation x̃0
        self.xr.value = np.asarray(xr, float)
        self.ur.value = np.asarray(ur, float)
        if self.u_prev is not None:
            if u_prev is None:
                self.u_prev.value = self.ur.value  # assume steady-state
            else:
                self.u_prev.value = np.asarray(u_prev, float)
        val = self.problem.solve(**self.solve_kwargs)
        if self.U.value is None:
            raise RuntimeError("MPC solve failed (U is None).")
        u_dev0 = np.asarray(self.U.value[:, 0]).reshape(-1)
        return self.ur.value + u_dev0

def dare_terminal_weight(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Standard discrete-time terminal weight from LQR∞."""
    return solve_discrete_are(Ad, Bd, Q, R)

def box_bounds_from_rpm(params, f_u2w2, f_w22u, rpm_min: float, rpm_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Conservative **box** bounds on u = [T, τx, τy, τz] from RPM bounds.
    Enumerates the 16 corners of Ω^2 ∈ {min^2, max^2}^4, maps through allocation.
    """
    import itertools
    w2_vals = [rpm_min**2, rpm_max**2]
    samples = []
    for combo in itertools.product(w2_vals, repeat=4):
        u = f_w22u(np.array(combo, float), params)
        samples.append(u)
    S = np.stack(samples, axis=1)  # shape (4, 16)
    umin = np.min(S, axis=1)
    umax = np.max(S, axis=1)
    return umin, umax



def main():
    pass


if __name__ == "__main__":
    main()
