from __future__ import annotations
"""
General-purpose utilities (math, integration, references, discretization).

"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Iterable
import numpy as np
import os, json as _json
import datetime as dt
import io
from contextlib import redirect_stdout, redirect_stderr
from macros import LIN_EPS_X  # shared finite-diff step

__all__ = [
    "sat", "wrap_pi", "finite_difference_jacobian",
    "R_zyx", "W_zyx",
    "rk4_step", "rollout",
    "make_Ttau_projector",
    "MinJerk1D", "make_minjerk3", "make_hover_policy_Ttau",
    "c2d_series",
    "make_policy_from_gain",
    "save_npz_package", "load_npz_package",
    "angle_peaks_deg",
]

# Basic math / utilities

def sat(x: np.ndarray, lo: float | np.ndarray, hi: float | np.ndarray) -> np.ndarray:
    """Elementwise saturation of x into [lo, hi]."""
    return np.minimum(np.maximum(x, lo), hi)

def wrap_pi(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to (-π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def finite_difference_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    eps: float = LIN_EPS_X
) -> np.ndarray:
    """Central-difference Jacobian J = ∂f/∂x|_{x0} for vector→vector maps."""
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(f(x0), dtype=float)
    ny, nx = y0.size, x0.size
    J = np.zeros((ny, nx), dtype=float)
    for i in range(nx):
        dx = np.zeros(nx); dx[i] = eps
        J[:, i] = (f(x0 + dx) - f(x0 - dx)) / (2.0 * eps)
    return J

# ZYX rotation & Euler-rate maps

def R_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body→inertial rotation for ZYX Euler angles: R = Rz(ψ) Ry(θ) Rx(φ)."""
    cφ, sφ = np.cos(phi), np.sin(phi)
    cθ, sθ = np.cos(theta), np.sin(theta)
    cψ, sψ = np.cos(psi), np.sin(psi)
    Rz = np.array([[ cψ, -sψ, 0.0],
                   [ sψ,  cψ, 0.0],
                   [ 0.0, 0.0, 1.0]])
    Ry = np.array([[ cθ, 0.0, sθ],
                   [ 0.0, 1.0, 0.0],
                   [-sθ, 0.0, cθ]])
    Rx = np.array([[ 1.0, 0.0, 0.0],
                   [ 0.0,  cφ, -sφ],
                   [ 0.0,  sφ,  cφ]])
    return Rz @ Ry @ Rx

def W_zyx(phi: float, theta: float) -> np.ndarray:
    r"""Euler-rate mapping for ZYX:
        [φ̇ θ̇ ψ̇]^T = W(φ,θ) [p q r]^T

    Uses a *safe* implementation of tan(θ) := sinθ / cosθ to avoid spikes
    when cosθ ≈ 0. This mirrors the internal PMP implementation.
    """
    cφ, sφ = np.cos(phi), np.sin(phi)
    cθ, sθ = np.cos(theta), np.sin(theta)
    # Guard near the Euler singularity at θ = ±90°
    if np.isclose(cθ, 0.0):
        cθ = 1e-12 if cθ >= 0.0 else -1e-12
    tanθ = sθ / cθ
    return np.array([
        [1.0, sφ * tanθ,  cφ * tanθ],
        [0.0, cφ,        -sφ       ],
        [0.0, sφ / cθ,    cφ / cθ  ]
    ])

# RK4 integrator & rollout

def rk4_step(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    dt: float
) -> np.ndarray:
    """One RK4 step for ẋ = f(x,u)."""
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rollout(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    policy: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    t0: float,
    tf: float,
    dt: float,
    project_u: Optional[Callable[[np.ndarray], Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Roll out dynamics with RK4 under a (t,x)↦u policy; returns (T, X)."""
    N = int(np.floor((tf - t0)/dt)) + 1
    nx = x0.size
    T = np.linspace(t0, t0 + (N-1)*dt, N)
    X = np.zeros((N, nx)); X[0] = x0
    x = x0.copy()
    for k in range(N-1):
        u = policy(T[k], x)
        if project_u is not None:
            pu = project_u(u)
            u = pu[0] if isinstance(pu, tuple) else pu
        x = rk4_step(f, x, u, dt)
        X[k+1] = x
    return T, X

# Input feasibility

def make_Ttau_projector(
    params: Any,
    thrust_torques_from_omegas2: Callable[[np.ndarray, Any], np.ndarray],
    omegas2_from_thrust_torques: Callable[[np.ndarray, Any], np.ndarray]
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Build a projector π(Tτ) that:
      1) maps Tτ → ω² via inverse allocation,
      2) clips ω² into [0, (max_rpm)^2],
      3) maps back to feasible Tτ via forward allocation.
    Returns (Tτ_feasible, ω²_clipped).
    """
    rpm2_max = params.max_rpm**2
    def proj(Ttau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        omega2 = omegas2_from_thrust_torques(np.asarray(Ttau, float), params)
        omega2 = sat(omega2, 0.0, rpm2_max)
        Ttau_feas = thrust_torques_from_omegas2(omega2, params)
        return Ttau_feas, omega2
    return proj

# References (min-jerk)

@dataclass
class MinJerk1D:
    """1D minimum-jerk: p(0)=p0, p(T)=pf, v(0)=v(T)=a(0)=a(T)=0."""
    p0: float
    pf: float
    T: float
    def eval(self, t: float) -> Tuple[float, float, float]:
        """Return (pos, vel, acc) at t∈[0,T]."""
        if self.T <= 0:
            return self.pf, 0.0, 0.0
        τ = np.clip(t, 0.0, self.T) / self.T
        τ2, τ3, τ4, τ5 = τ*τ, τ**3, τ**4, τ**5
        Δ = self.pf - self.p0
        p = self.p0 + Δ * (10*τ3 - 15*τ4 + 6*τ5)
        v = (Δ/self.T) * (30*τ2 - 60*τ3 + 30*τ4)
        a = (Δ/self.T**2) * (60*τ - 180*τ2 + 120*τ3)
        return float(p), float(v), float(a)

def make_minjerk3(p0: np.ndarray, pf: np.ndarray, T: float
) -> Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    mj = [MinJerk1D(float(p0[i]), float(pf[i]), T) for i in range(3)]
    def ref(t: float):
        pv = [mj[i].eval(t) for i in range(3)]
        p = np.array([pv[i][0] for i in range(3)], float)
        v = np.array([pv[i][1] for i in range(3)], float)
        a = np.array([pv[i][2] for i in range(3)], float)
        return p, v, a
    return ref

def make_hover_policy_Ttau(u0: np.ndarray) -> Callable[[float, np.ndarray], np.ndarray]:
    """Constant hover-input policy (returns [T0, 0, 0, 0])."""
    u0 = np.asarray(u0, float).copy()
    def π(t: float, x: np.ndarray) -> np.ndarray:
        return u0
    return π

# Continuous → Discrete (ZOH, series)

def c2d_series(A: np.ndarray, B: np.ndarray, dt: float, order: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    ZOH discretization using truncated series (adequate for small dt).
    A_d ≈ Σ (A dt)^k/k!,  B_d ≈ Σ (A^k dt^{k+1}/(k+1)!) B
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    nx, nu = B.shape
    Ad = np.eye(nx); Ak = np.eye(nx)
    Bd = np.zeros((nx, nu))
    for k in range(1, order+1):
        Ak = Ak @ (A * (dt / k))
        Ad = Ad + Ak
    Ak = np.eye(nx)
    for k in range(0, order+1):
        coef = (dt**(k+1)) / np.math.factorial(k+1)
        if k > 0:
            Ak = Ak @ A
        Bd = Bd + coef * (Ak @ B)
    return Ad, Bd

# Linear feedback policy

def make_policy_from_gain(
    K: np.ndarray, x_eq: np.ndarray, u_eq: np.ndarray,
    project_u: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Deviation-form linear feedback: u(x) = u_eq - K (x - x_eq)."""
    K = np.asarray(K, float); x_eq = np.asarray(x_eq, float); u_eq = np.asarray(u_eq, float)
    def π(t: float, x: np.ndarray) -> np.ndarray:
        u = u_eq - K @ (x - x_eq)
        if project_u is not None:
            u, _ = project_u(u)
        return u
    return π

# NPZ save/load

def save_npz_package(path: str, arrays: dict, meta: dict | None = None) -> str:
    """Save dict of NumPy arrays into compressed .npz with a JSON meta block."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    meta = {} if meta is None else dict(meta)
    meta.setdefault("timestamp", dt.datetime.now().isoformat(timespec="seconds"))
    meta_json = _json.dumps(meta, indent=2)
    np.savez_compressed(path, **payload, meta_json=np.array(meta_json))
    return path

def load_npz_package(path: str) -> tuple[dict, dict]:
    """Load arrays + meta dict saved by `save_npz_package`."""
    data = np.load(path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files if k != "meta_json"}
    meta = {}
    if "meta_json" in data.files:
        try:
            meta = _json.loads(str(data["meta_json"]))
        except Exception:
            meta = {}
    return arrays, meta

import numpy as _np

def load_lqr_artifact(artifacts_dir: str, prefix: str, path: str | None = None):
    """
    Load an LQR artifact npz.
    If 'path' is None, picks newest artifacts_dir/prefix*.npz.
    Returns (path, arrays_dict, meta_dict).
    """
    if path is None:
        from macros import newest_path  # local import to avoid cycles at import time
        path = newest_path(artifacts_dir, prefix, ".npz")
        if path is None:
            raise FileNotFoundError("No LQR artifact found. Run lqr.py first.")
    data = _np.load(path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files if k != "meta_json"}
    meta = {}
    if "meta_json" in data.files:
        meta = _json.loads(str(data["meta_json"]))
    return path, arrays, meta

def rk4_step_lin(A: _np.ndarray, B: _np.ndarray, x: _np.ndarray, u: _np.ndarray, dt: float) -> _np.ndarray:
    """RK4 step for linear system xdot = A x + B u."""
    k1 = A @ x + B @ u
    k2 = A @ (x + 0.5*dt*k1) + B @ u
    k3 = A @ (x + 0.5*dt*k2) + B @ u
    k4 = A @ (x + dt*k3)     + B @ u
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def apply_rpm_saturation(
    u_dev: _np.ndarray, u_eq: _np.ndarray, params,
    f_u2w2, f_w22u, clip_fun
):
    """
    Saturate deviation input in RPM space using allocation:
      u_abs = u_eq + u_dev -> omega^2 -> omega (clip) -> u_abs_feas -> u_dev_feas.
    Returns (u_dev_feas, omega_RPM).
    Callers pass allocation funcs from dynamics: (omegas2_from_thrust_torques, thrust_torques_from_omegas2, clip_omegas).
    """
    u_abs = u_eq + u_dev
    omega2 = f_u2w2(u_abs, params)
    omega  = _np.sqrt(_np.maximum(omega2, 0.0))
    omega  = clip_fun(omega, params)
    u_abs_feas = f_w22u(omega**2, params)
    return (u_abs_feas - u_eq), omega

def ct_cost(T: _np.ndarray, X: _np.ndarray, U: _np.ndarray, Q: _np.ndarray, R: _np.ndarray, Xref: _np.ndarray) -> float:
    """Continuous-time LQR cost ∫ 0.5(x'Qx + u'Ru) dt (trapezoid)."""
    E = X - Xref
    xQx = _np.einsum("ij,ni,nj->n", Q, E, E)
    uRu = _np.einsum("ij,ni,nj->n", R, U, U)
    return float(_np.trapz(0.5*(xQx + uRu), T))

def step_settling_time(T: np.ndarray, y: np.ndarray, y_ref: float, tol: float = 0.02, eps: float = 1e-9) -> float:
    """
    Settling time for a step response y(t) relative to constant y_ref.
    Returns the first time t such that |y(t') - y_ref| <= band for all t' >= t,
    where band = tol*|Δ| with Δ = y_ref - y[0]. If |Δ| ~ 0 (zero-reference case),
    fall back to an absolute band ±tol. If never settles, returns np.nan.
    """
    T = np.asarray(T, float)
    y = np.asarray(y, float)

    amp = float(abs(y_ref - y[0]))
    # If the step amplitude is ~0 (e.g., phi/theta around zero), interpret `tol` as absolute.
    band = tol if amp <= eps else tol * amp

    inside = np.abs(y - y_ref) <= band
    if not np.any(inside):
        return np.nan

    # First index i such that inside[i:] are all True (suffix-AND)
    for i in range(len(y)):
        if inside[i] and inside[i:].all():
            return float(T[i])
    return np.nan

def step_overshoot_pct(T, y, y_ref, eps: float = 1e-9) -> float:
    """
    Classic percent overshoot for a step from y[0] to y_ref:
      OS% = 100 * max(0, excursion past y_ref in the step direction) / |y_ref - y[0]|.
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    # Coerce y_ref to scalar robustly
    yr = np.asarray(y_ref)
    if yr.ndim == 0 or yr.size == 1:
        yref = float(yr.reshape(-1)[0])
    elif yr.shape == y.shape:
        yref = float(yr[-1])
    else:
        yref = float(np.ravel(yr)[0])

    # Step amplitude
    amp = yref - y[0]
    if abs(amp) <= eps:
        return 0.0  # undefined for zero step; return 0 by convention

    if amp > 0.0:
        excursion = float(np.max(y) - yref)   # upward step: overshoot above y_ref
    else:
        excursion = float(yref - np.min(y))   # downward step: undershoot below y_ref

    return 100.0 * max(0.0, excursion) / abs(amp)

def step_metrics_multi(
    T: np.ndarray, X: np.ndarray, xr: np.ndarray,
    indices: Iterable[int] = (0, 1, 2, 8), tol: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute settling time and percent overshoot for multiple state indices
    relative to constant reference xr (per-state constant).
    Returns (indices_array, settling_times [s], overshoot_pct [%]).
    If a signal never settles, settling time is np.nan.
    """
    T = np.asarray(T, float)
    X = np.asarray(X, float)
    xr = np.asarray(xr, float)
    idxs = np.array(list(indices), dtype=int)
    ts = np.zeros_like(idxs, dtype=float)
    os = np.zeros_like(idxs, dtype=float)
    for k, i in enumerate(idxs):
        y = X[:, i]
        yref = float(xr[i])
        ts[k] = step_settling_time(T, y, yref, tol=tol)
        os[k] = step_overshoot_pct(T, y, yref)
    return idxs, ts, os

def peak_abs_deflection(y: np.ndarray, y_ref: float = 0.0) -> float:
    """
    Absolute peak deviation from y_ref (same units as y).
    Returns a scalar (e.g., radians for attitude).
    """
    y = np.asarray(y, float).reshape(-1)
    return float(np.max(np.abs(y - y_ref)))

def _scalar_ref(y_ref, y):
    """Robustly coerce y_ref (scalar/array) to a scalar reference for this signal."""
    yr = np.asarray(y_ref)
    if yr.ndim == 0 or yr.size == 1:
        return float(yr.reshape(-1)[0])
    if yr.shape == np.asarray(y).shape:
        return float(yr[-1])  # time-varying ref: use final value
    return float(np.ravel(yr)[0])

def angle_peaks_deg(X: np.ndarray, xr: np.ndarray) -> np.ndarray:
    """
    Peak |phi|, |theta|, and |psi-psi_ref| in DEGREES with angles wrapped to (-pi, pi].
    Returns np.array([phi_pk_deg, theta_pk_deg, psi_rel_pk_deg]).
    """
    X = np.asarray(X, float)
    phi_pk   = float(np.max(np.abs(wrap_pi(X[:, 6]))))
    theta_pk = float(np.max(np.abs(wrap_pi(X[:, 7]))))
    psi_pk   = float(np.max(np.abs(wrap_pi(X[:, 8] - float(np.asarray(xr)[8])))))
    return np.degrees(np.array([phi_pk, theta_pk, psi_pk], dtype=float))

# NPZ save/load
class Tee(io.TextIOBase):
    """Write to multiple streams (console + file)."""
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
        return len(s)
    def flush(self):
        for st in self.streams: st.flush()




def main():
    pass


if __name__ == "__main__":
    main()
