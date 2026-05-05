#!/usr/bin/env python3
from __future__ import annotations
"""
mpc_nonlinear.py
Nonlinear MPC (multiple-shooting, explicit RK4)
(casadi_ms.make_symbolic_dynamics).

- Decision var: W2 = Ω^2 per rotor (RPM^2) with per-rotor bounds.
- Cost on absolute input: u = [T, τx, τy, τz] = M @ W2 (tracks UR).
- Optional rate bound: |M (W2_k - W2_{k-1})| ≤ du_max (in Tτ space).
- IPOPT options flattened as 'ipopt.*' for CasADi compatibility.

API:
    cfg = NLMPCConfig(N, dt, Q, R, Qf, rpm_min, rpm_max, du_max, ...)
    nmpc = NonlinearMPC_MS(params, cfg)
    nmpc.set_params(x0, XR, UR, w2_prev)   # (12,), (12,N+1), (4,N), (4,)
    sol = nmpc.solve(warm={"X": X_guess, "W2": W2_guess})
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import casadi as ca

from casadi_ms import make_symbolic_dynamics

@dataclass
class NLMPCConfig:
    N: int
    dt: float
    Q: np.ndarray
    R: np.ndarray
    Qf: Optional[np.ndarray] = None
    rpm_min: float = 0.0
    rpm_max: float = 1e6
    du_max: Optional[np.ndarray] = None   # in Tτ space, shape (4,)
    ipopt_max_iter: int = 500
    ipopt_tol: float = 1e-4
    verbose: bool = False

    def __post_init__(self):
        if self.Qf is None:
            self.Qf = np.array(self.Q, copy=True)

class NonlinearMPC_MS:
    """NMPC with multiple shooting and RK4 via Φ(x,W2;h)."""

    def __init__(self, params, cfg: NLMPCConfig):
        self.params = params
        self.cfg = cfg
        self._build_problem()

    def _build_problem(self):
        params = self.params
        cfg = self.cfg
        nx, nu, N = 12, 4, cfg.N

        # symbolic step and allocation
        Phi, _PhiJ, M = make_symbolic_dynamics(params)   # Φ(x,W2,h)->(xnext, Tτ)
        self.M = M

        opti = ca.Opti()

        # Decision variables
        X  = opti.variable(nx, N+1)
        W2 = opti.variable(nu, N)   # RPM^2

        # Parameters
        x0      = opti.parameter(nx, 1)
        XR      = opti.parameter(nx, N+1)
        UR      = opti.parameter(nu, N)
        W2_PREV = opti.parameter(nu, 1)

        # Rotor bounds (RPM^2)
        w2_min = float(cfg.rpm_min)**2
        w2_max = float(cfg.rpm_max)**2

        # Dynamics & bounds
        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            xnext, _u_k = Phi(X[:, k], W2[:, k], cfg.dt)   # MX-safe call into SX-built Function
            opti.subject_to(X[:, k+1] == xnext)
            opti.subject_to(opti.bounded(w2_min, W2[:, [k]], w2_max))

        # Optional |Δu| bound in Tτ space
        if cfg.du_max is not None:
            du = ca.DM(np.asarray(cfg.du_max, float)).reshape((nu,1))
            U0    = M @ W2[:, [0]]
            Uprev = M @ W2_PREV
            opti.subject_to(ca.fabs(U0 - Uprev) <= du)
            for k in range(1, N):
                Uk   = M @ W2[:, [k]]
                Ukm1 = M @ W2[:, [k-1]]
                opti.subject_to(ca.fabs(Uk - Ukm1) <= du)

        # Quadratic tracking cost
        Q, R, Qf = ca.DM(cfg.Q), ca.DM(cfg.R), ca.DM(cfg.Qf)
        cost = 0
        for k in range(N):
            dx = X[:, [k]] - XR[:, [k]]
            uk = M @ W2[:, [k]]
            du = uk - UR[:, [k]]
            cost += (ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])) * cfg.dt
        dxN = X[:, [N]] - XR[:, [N]]
        cost += ca.mtimes([dxN.T, Qf, dxN])
        opti.minimize(cost)

        # IPOPT (flattened keys; avoids the 'No such IPOPT option: ipopt' issue)
        opts = {
            "expand": True,
            "print_time": cfg.verbose,
            "ipopt.max_iter": int(cfg.ipopt_max_iter),
            "ipopt.tol": float(cfg.ipopt_tol),
            "ipopt.print_level": 5 if cfg.verbose else 0,
            "ipopt.sb": "yes" if not cfg.verbose else "no",
        }
        opti.solver("ipopt", opts)

        # Store handles
        self.opti = opti
        self.X, self.W2 = X, W2
        self.x0, self.XR, self.UR, self.W2_PREV = x0, XR, UR, W2_PREV
        self.nx, self.nu, self.N = nx, nu, N

    def set_params(self, x0: np.ndarray, xr_seq: np.ndarray, ur_seq: np.ndarray, w2_prev: np.ndarray):
        x0      = np.asarray(x0, float).reshape((-1,1))
        xr_seq  = np.asarray(xr_seq, float).reshape((self.nx, self.N+1))
        ur_seq  = np.asarray(ur_seq, float).reshape((self.nu, self.N))
        w2_prev = np.asarray(w2_prev, float).reshape((-1,1))
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.XR, xr_seq)
        self.opti.set_value(self.UR, ur_seq)
        self.opti.set_value(self.W2_PREV, w2_prev)

    def solve(self, warm: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        if warm is not None:
            if "X" in warm and warm["X"] is not None:
                self.opti.set_initial(self.X, warm["X"])
            if "W2" in warm and warm["W2"] is not None:
                self.opti.set_initial(self.W2, warm["W2"])
        try:
            sol = self.opti.solve()
            X  = sol.value(self.X)
            W2 = sol.value(self.W2)
            U  = (self.M @ W2).full()
            O  = np.sqrt(np.maximum(W2, 0.0))  # RPM
            return {"status": "solved", "X": X, "U": U, "W2": W2, "O": O, "cost": float(sol.value(self.opti.f))}
        except RuntimeError as e:
            try:
                X  = self.opti.debug.value(self.X)
                W2 = self.opti.debug.value(self.W2)
                U  = (self.M @ W2).full()
                O  = np.sqrt(np.maximum(W2, 0.0))
            except Exception:
                X = W2 = U = O = None
            return {"status": f"failed: {e}", "X": X, "U": U, "W2": W2, "O": O, "cost": np.inf}



def main():
    pass


if __name__ == "__main__":
    main()
