from __future__ import annotations

"""LQR design utilities (infinite- and finite-horizon) + demo.

This module provides:
- LinearModel dataclass and a `build_linear_hover_model` convenience wrapper.
- Robust CARE solver (SciPy first, NumPy Hamiltonian fallback).
- Finite-horizon LQR via backward integration of the DRE (stiff solver).
- Time-varying LQR rollout using precomputed schedules (Pontryagin form).
- A demo `__main__` section that designs controllers and produces figures.

"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import sys, os, datetime as dt
from dynamics import QuadParams, linearize_hover
from plotting import apply_style, heatmap, state_labels, input_labels, save_fig  # (labels defined locally below)
from utils import save_npz_package, Tee, redirect_stderr, redirect_stdout
from macros import ARTIFACTS_DIR, LQR_PREFIX, Q_matrix, R_matrix

# Linear model container

@dataclass
class LinearModel:
    A: np.ndarray
    B: np.ndarray
    x0: np.ndarray
    u0: np.ndarray
    desc: str = "Hover linearization (inputs: [T, τx, τy, τz])"

# Controllability

def build_linear_hover_model(params: Optional[QuadParams] = None, yaw: float = 0.0) -> LinearModel:
    """Construct (A,B,x0,u0) around hover using dynamics.linearize_hover."""
    if params is None:
        params = QuadParams()
    A, B, x0, u0 = linearize_hover(params, yaw=yaw)
    return LinearModel(A=A, B=B, x0=x0, u0=u0)

def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    C = [B, A B, A^2 B, ..., A^{n-1} B]  ∈ R^{n × (n m)}
    where n = A.shape[0], m = B.shape[1].
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    n, m = A.shape[0], B.shape[1]
    C = np.empty((n, n * m), dtype=float)
    AkB = B.copy()
    for i in range(n):
        C[:, i*m:(i+1)*m] = AkB
        AkB = A @ AkB
    return C

def is_stabilizable(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> bool:
    """
    PBH stabilizability: (A,B) stabilizable iff rank([λI-A, B]) = n
    for every eigenvalue λ with Re(λ) ≥ 0 (within tol).
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    n = A.shape[0]
    evals = np.linalg.eigvals(A)
    for lam in evals:
        if lam.real >= -tol:
            M = np.hstack((lam*np.eye(n) - A, B))
            if np.linalg.matrix_rank(M) < n:
                return False
    return True

# CARE / LQR (continuous-time)

def _care_scipy(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.linalg import solve_continuous_are
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return P, K

def _care_hamiltonian(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy-only CARE via Hamiltonian eigenspace."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    Q = np.asarray(Q, float); R = np.asarray(R, float)
    R_sym = 0.5 * (R + R.T)
    Rinv  = np.linalg.inv(R_sym + 1e-12*np.eye(R.shape[0]))
    H = np.block([[ A,              -B @ Rinv @ B.T],
                  [-Q,              -A.T           ]])
    ew, V = np.linalg.eig(H)
    sel = ew.real < 0.0
    n = A.shape[0]
    Vs = V[:n, sel]; Us = V[n:, sel]
    P  = Us @ np.linalg.inv(Vs)
    P  = 0.5 * (P + P.T)
    K  = Rinv @ (B.T @ P)
    return P, K

def solve_care(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        return _care_scipy(A, B, Q, R)
    except Exception:
        return _care_hamiltonian(A, B, Q, R)

# Finite-horizon LQR (robust backward DRE integration)

def finite_horizon_lqr(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, S: np.ndarray,
    tf: float, nsamp: int = 201, opts: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    -Ṗ = AᵀP + PA - PBR^{-1}BᵀP + Q,  P(T)=S
    Returns t, P_seq(t), K_seq(t)=R^{-1}BᵀP(t)
    """
    from scipy.integrate import solve_ivp
    if opts is None: opts = {}
    rtol   = float(opts.get("rtol", 1e-8))
    atol   = float(opts.get("atol", 1e-10))
    method = str(opts.get("method", "BDF"))
    eps_psd = float(opts.get("eps_psd", 1e-10))
    max_fro = float(opts.get("max_fro", 1e6))

    A = np.asarray(A, float); B = np.asarray(B, float)
    Q = np.asarray(Q, float); R = np.asarray(R, float); S = np.asarray(S, float)
    n = A.shape[0]; m = B.shape[1]

    cF  = cho_factor(R, lower=True, check_finite=False)
    RBt = cho_solve(cF, B.T, check_finite=False)  # m×n
    G   = B @ RBt                                  # n×n

    def dre_rhs(_t, y):
        P = y.reshape(n, n)
        F = A.T @ P + P @ A - (P @ G @ P) + Q
        return (-F).reshape(-1)

    t_eval_desc = np.linspace(tf, 0.0, nsamp)

    S_sym = 0.5*(S + S.T)
    ew = np.linalg.eigvalsh(S_sym)
    if ew.min() < eps_psd:
        S_sym += (eps_psd - ew.min()) * np.eye(n)
    yT = S_sym.reshape(-1)

    sol = solve_ivp(dre_rhs, (tf, 0.0), yT, t_eval=t_eval_desc,
                    method=method, rtol=rtol, atol=atol, vectorized=False)
    if not sol.success:
        alt = "Radau" if method == "BDF" else "BDF"
        sol = solve_ivp(dre_rhs, (tf, 0.0), yT, t_eval=t_eval_desc,
                        method=alt, rtol=rtol, atol=atol, vectorized=False)
        if not sol.success:
            raise RuntimeError(f"DRE integration failed: {sol.message}")

    Pseq = sol.y.T.reshape(nsamp, n, n)[::-1].copy()  # ascending time
    for k in range(nsamp):
        Pk = 0.5*(Pseq[k] + Pseq[k].T)
        ew, V = np.linalg.eigh(Pk)
        ew = np.maximum(ew, eps_psd)
        Pseq[k] = (V * ew) @ V.T
        fro = np.linalg.norm(Pseq[k], ord="fro")
        if fro > max_fro:
            Pseq[k] *= (max_fro / fro)

    Kseq = np.zeros((nsamp, m, n))
    for k in range(nsamp):
        Kseq[k] = RBt @ Pseq[k]

    t = np.linspace(0.0, tf, nsamp)
    return t, Pseq, Kseq

# LQR design wrapper

def design_lqr(
    A: np.ndarray, B: np.ndarray,
    Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None,
    traj: bool = False, T_traj: float = 10.0, dt_traj: float = 0.02,
    S_mode: str = "scaleQ", S_scale: float = 5.0, S_custom: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns {Q,R,P,K,eig_ol,eig_cl,Ctrb,rank_ctrb, [t_fh,P_seq,K_seq,S_term]}.
    """
    nx, _ = B.shape
    Q = Q_matrix() if Q is None else np.asarray(Q, float)
    R = R_matrix() if R is None else np.asarray(R, float)

    Ctrb = controllability_matrix(A, B)
    rank_ctrb = int(np.linalg.matrix_rank(Ctrb))
    if rank_ctrb < nx and not is_stabilizable(A, B):
        raise RuntimeError(f"(A,B) is not stabilizable (rankCtrb={rank_ctrb}/{nx}); LQR design is not well-defined.")

    P_inf, K_inf = solve_care(A, B, Q, R)
    ol_eigs = np.linalg.eigvals(A)
    cl_eigs = np.linalg.eigvals(A - B @ K_inf)

    out = {"Q": Q, "R": R, "P": P_inf, "K": K_inf,
           "eig_ol": ol_eigs, "eig_cl": cl_eigs,
           "Ctrb": Ctrb, "rank_ctrb": rank_ctrb}

    if not traj:
        return out

    if S_mode == "Pinf":
        S_term = P_inf
    elif S_mode == "custom" and S_custom is not None:
        S_term = S_custom
    else:
        S_term = float(S_scale) * Q

    nsamp = int(np.floor(T_traj / dt_traj)) + 1
    t_fh, P_seq, K_seq = finite_horizon_lqr(
        A, B, Q, R, S_term, tf=T_traj, nsamp=nsamp,
        opts={"method": "BDF", "rtol": 1e-8, "atol": 1e-10, "eps_psd": 1e-10}
    )
    out.update({"t_fh": t_fh, "P_seq": P_seq, "K_seq": K_seq, "S_term": S_term})
    return out

def save_lqr_results_npz(
    path: str,
    A: np.ndarray, B: np.ndarray, x0: np.ndarray, u0: np.ndarray,
    lqr: dict, meta: dict | None = None
) -> str:
    """Persist LQR artifacts for later reuse (controller, weights, schedules)."""
    arrays = {"A": A, "B": B, "x0": x0, "u0": u0,
              "Q": lqr["Q"], "R": lqr["R"], "P": lqr["P"], "K": lqr["K"],
              "eig_ol": lqr["eig_ol"], "eig_cl": lqr["eig_cl"], "Ctrb": lqr["Ctrb"]}
    if "t_fh" in lqr: arrays["t_fh"] = lqr["t_fh"]
    if "P_seq" in lqr: arrays["P_seq"] = lqr["P_seq"]
    if "K_seq" in lqr: arrays["K_seq"] = lqr["K_seq"]
    if "S_term" in lqr: arrays["S_term"] = lqr["S_term"]
    meta = {} if meta is None else dict(meta)
    meta.setdefault("rank_ctrb", int(lqr.get("rank_ctrb", -1)))
    meta.setdefault("desc", "LQR around hover (inputs: [T, τx, τy, τz])")
    return save_npz_package(path, arrays, meta)

# Finite-horizon linearized PMP rollout (uses precomputed K,P)

def finite_horizon_pmp(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, S: np.ndarray,
    x0: np.ndarray, T: float, dt: float,
    t_sched: Optional[np.ndarray] = None, P_seq: Optional[np.ndarray] = None, K_seq: Optional[np.ndarray] = None,
    xref: Optional[callable] = None, interp: str = "pc", compute_cost: bool = True
) -> Dict[str, np.ndarray]:
    """
    PMP with u*(t) = -K(t)(x - x_ref(t)), λ(t) = P(t)(x - x_ref(t)).
    Uses precomputed schedules {t_sched, P_seq, K_seq}.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    Q = np.asarray(Q, float); R = np.asarray(R, float); S = np.asarray(S, float)
    n = A.shape[0]; m = B.shape[1]
    assert t_sched is not None and P_seq is not None and K_seq is not None, "Provide schedules from design_lqr(..., traj=True)."
    assert P_seq.shape[0] == t_sched.size and K_seq.shape[0] == t_sched.size
    assert P_seq.shape[1:] == (n, n) and K_seq.shape[1:] == (m, n)
    assert np.allclose(P_seq[-1], 0.5*(S+S.T), atol=1e-8), "P(T) must equal S"

    if xref is None:
        def xref(_t: float) -> np.ndarray:
            return np.zeros(n)

    N = int(np.floor(T / dt)) + 1
    t = np.linspace(0.0, T, N)

    if interp.lower() == "pc":
        idx = np.searchsorted(t_sched, t[:-1], side="right") - 1
        idx = np.clip(idx, 0, t_sched.size - 1)
        def Kk_at(k: int): return K_seq[idx[k]]
        def Pk_at(k: int): return P_seq[idx[k]]
        K_last, P_last = K_seq[-1], P_seq[-1]
    elif interp.lower() == "linear":
        jhi = np.searchsorted(t_sched, t[:-1], side="right")
        jlo = np.clip(jhi - 1, 0, t_sched.size - 2)
        jhi = np.clip(jhi,       1, t_sched.size - 1)
        t0 = t_sched[jlo]; t1 = t_sched[jhi]
        w1 = (t[:-1] - t0) / np.maximum(t1 - t0, 1e-12); w0 = 1.0 - w1
        def Kk_at(k: int): return w0[k] * K_seq[jlo[k]] + w1[k] * K_seq[jhi[k]]
        def Pk_at(k: int): return w0[k] * P_seq[jlo[k]] + w1[k] * P_seq[jhi[k]]
        K_last, P_last = K_seq[-1], P_seq[-1]
    else:
        raise ValueError("interp must be 'pc' or 'linear'")

    def rk4_step_lin(x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
        k1 = A @ x + B @ u
        k2 = A @ (x + 0.5*h*k1) + B @ u
        k3 = A @ (x + 0.5*h*k2) + B @ u
        k4 = A @ (x + h*k3)     + B @ u
        return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    X = np.zeros((N, n)); U = np.zeros((N, m)); Lmb = np.zeros((N, n))
    X[0] = x0
    for k in range(N - 1):
        xr = xref(t[k]); ek = X[k] - xr
        Kk = Kk_at(k);  Pk = Pk_at(k)
        U[k] = -Kk @ ek
        X[k+1] = rk4_step_lin(X[k], U[k], dt)
        Lmb[k] = Pk @ ek
    eT = X[-1] - xref(t[-1])
    U[-1]  = -K_last @ eT
    Lmb[-1] = P_last @ eT

    J = Jstage = Jterm = None
    if compute_cost:
        Xref = np.stack([xref(tk) for tk in t], axis=0)
        E = X - Xref
        xQx = np.einsum('ij,ni,nj->n', Q, E, E)
        uRu = np.einsum('ij,nj,ni->n', R, U, U)
        Jstage = np.trapz(0.5*(xQx + uRu), t)
        Jterm  = 0.5 * (E[-1].T @ S @ E[-1])
        J = Jstage + Jterm

    return {"t": t, "X": X, "U": U, "lambda": Lmb,
            "J": J, "stage_cost": Jstage, "terminal_cost": Jterm}

# Main
if __name__ == "__main__":
    # Timestamp + paths (shared for artifact, figures, and log)
    stamp    = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp    = '' # overwrite everytime
    npz_path = os.path.join(ARTIFACTS_DIR, f"{LQR_PREFIX}{stamp}.npz")
    log_path = os.path.join(ARTIFACTS_DIR, f"{LQR_PREFIX}{stamp}_report.txt")

    # Ensure the artifacts directory exists BEFORE opening the log
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    yaw0    = 0.0
    T_traj  = 12.0      # seconds of schedule for P(t), K(t)
    dt_traj = 0.02
    S_mode  = "scaleQ"  # "scaleQ" | "Pinf" | "custom"
    S_scale = 5.0

    apply_style(theme="latex", context="paper", font_scale=1.0)

    params = QuadParams()
    lm = build_linear_hover_model(params, yaw=yaw0)
    Q, R = Q_matrix(), R_matrix()
    lqr = design_lqr(
        lm.A, lm.B, Q=Q, R=R,
        traj=True, T_traj=T_traj, dt_traj=dt_traj,
        S_mode=S_mode, S_scale=S_scale
    )

    def _eig_table(eigs: np.ndarray, title: str) -> str:
        vals = np.asarray(eigs, complex).copy()
        vals = vals[np.argsort(vals.real)]
        header = f"{title}\n" + "-"*len(title) + \
                 "\n  idx           Re(lambda)        Im(lambda)"
        rows = []
        for i, lam in enumerate(vals):
            rows.append(f"{i:5d}  {lam.real:14.6e}  {lam.imag:14.6e}")
        return header + "\n" + "\n".join(rows)

    def _print_matrix(name: str, M: np.ndarray):
        print(f"\n{name}  shape={M.shape}")
        print(M)

    # NOTE: Open the log as UTF-8 (safe), but keep console prints ASCII.
    with open(log_path, "w", encoding="utf-8") as _log, \
         redirect_stdout(Tee(sys.stdout, _log)), \
         redirect_stderr(Tee(sys.stderr, _log)):

        print("\n=== LQR design at hover ===")

        # Controllability
        Ctrb = lqr["Ctrb"]; rank_ctrb = int(lqr["rank_ctrb"])
        print(f"Controllability: rank(C) = {rank_ctrb} / {lm.A.shape[0]}")
        _print_matrix("C (controllability matrix)", Ctrb)

        # Eigenvalues
        print("\n" + _eig_table(lqr["eig_ol"], "Open-loop eigenvalues (A)"))
        print("\n" + _eig_table(lqr["eig_cl"], "Closed-loop eigenvalues (A - B K_inf)"))

        # K_inf and P_inf (+ PD checks)
        K = lqr["K"]; P = lqr["P"]
        _print_matrix("K_inf", K)
        _print_matrix("P_inf", P)
        Psym = 0.5 * (P + P.T)
        lamP = np.linalg.eigvalsh(Psym)
        print(f"\nP is positive definite? {np.all(lamP > 0)}  (lambda_min = {lamP.min():.3e})")
        np.linalg.cholesky(Psym); print("P is PD (Cholesky): True")

        # Save artifact (after reporting so paths are in the log)
        save_lqr_results_npz(
            npz_path, lm.A, lm.B, lm.x0, lm.u0, lqr,
            meta={
                "desc": "Hover LQR (inputs: [T, tau_x, tau_y, tau_z])",
                "weights": {"Q_diag": np.diag(Q).tolist(), "R_diag": np.diag(R).tolist()},
                "yaw0": float(yaw0),
            },
        )
        print(f"\nSaved LQR artifact -> {npz_path}")

        s_labels = state_labels()
        u_labels = input_labels()

        # (1) P(t) diagonals
        tP, P_seq = lqr["t_fh"], lqr["P_seq"]    # (N, n, n)
        figP, axP = plt.subplots(figsize=(10, 5))
        for i, lab in enumerate(s_labels):
            axP.plot(tP, P_seq[:, i, i], label=lab, linewidth=1.4)
        axP.set_title(r"Finite-horizon $P(t)$ diagonals")
        axP.set_xlabel("time [s]"); axP.set_ylabel(r"$P_{ii}(t)$")
        axP.legend(ncol=4)
        pfig_path = os.path.join(ARTIFACTS_DIR, f"{LQR_PREFIX}{stamp}_P_diagonals.pdf")
        save_fig(figP, pfig_path); plt.close(figP)
        print(f"Figure saved -> {pfig_path}")

        # (2) K(t) — 12 curves for each input row (four figures)
        K_seq = lqr["K_seq"]                    # (N, m, n)
        _, m, n = K_seq.shape
        for i in range(m):
            u_lab = u_labels[i]
            figK, axK = plt.subplots(figsize=(11, 6))
            for j, s_lab in enumerate(s_labels):
                axK.plot(tP, K_seq[:, i, j], linewidth=1.2, label=fr"{u_lab}·{s_lab}")
            axK.set_title(rf"Finite-horizon $K_{{{u_lab[1:-1]},*}}(t)$ - 12 gains")
            axK.set_xlabel("time [s]"); axK.set_ylabel("gain")
            axK.legend(ncol=4)
            kfig_path = os.path.join(ARTIFACTS_DIR, f"{LQR_PREFIX}{stamp}_K_row_{i}.pdf")
            save_fig(figK, kfig_path); plt.close(figK)
            print(f"Figure saved -> {kfig_path}")

        print(f"\nFull log written to -> {log_path}\n")
