#!/usr/bin/env python3
from __future__ import annotations
"""
view_saved.py
Simple viewer for LQR artifacts and simulation NPZs.

- Set ARTIFACT_PATHS and/or SIM_PATHS below (leave empty to skip).
- Artifact: prints shapes, controllability rank, eigenvalues, PD(P); plots P(t) diagonals and K(t) per input if present.
- Simulation: detects linear vs nonlinear by keys, prints cost + saturation, and plots states stack + RPMs.
- Shows figures interactively.
"""

import os, numpy as np, matplotlib.pyplot as plt
from datetime import datetime

from macros import ARTIFACTS_DIR, SIM_DIR, LQR_PREFIX
from plotting import (
    apply_style,
    state_labels,
    _plot_states_2x2,
    _plot_rpms,
    _plot_costates_stack,
    _plot_pmp_residuals_inf,
)
from dynamics import QuadParams
from utils import load_lqr_artifact, wrap_pi

ARTIFACT_PATHS = [os.path.join(ARTIFACTS_DIR, "lqr_hover_.npz")]
SIM_PATHS = [
    os.path.join(SIM_DIR, "lqr_lin_.npz"),
    os.path.join(SIM_DIR, "lqr_nonlin_.npz"),
    os.path.join(SIM_DIR, "mpc_lin_.npz"),
    os.path.join(SIM_DIR, "mpc_nonlin_.npz"),
    os.path.join(SIM_DIR, "ocp_nonlin_ms_casadi_.npz"),
]

PLOT = False
SAVE = True
def _print_mpc_params(npz, keys: set[str]):
    # Kind
    if "Ad" in keys or "Bd" in keys:
        kind = "Linear MPC"
    elif ("dt_ctrl" in keys) and ("N_horizon" in keys):
        kind = "Nonlinear MPC"
    else:
        kind = "MPC (unspecified)"

    # Time steps & horizon
    dt_ctrl = npz.get("dt_ctrl", None)
    N_hor   = npz.get("N_horizon", None)
    dt_ctrl_s = f"{float(dt_ctrl):.4f}s" if dt_ctrl is not None else "— (not saved)"
    N_s       = f"{int(N_hor)}"          if N_hor   is not None else "— (not saved)"

    # Weights
    Q = npz.get("Q", None); R = npz.get("R", None); P = npz.get("P", None)
    Ad = npz.get("Ad",None); Bd = npz.get("Bd",None)

    # Bounds (if saved)
    rpm_min = npz.get("rpm_min", None); rpm_max = npz.get("rpm_max", None)
    umin = npz.get("umin", None); umax = npz.get("umax", None)

    print("\n[MPC params]")
    print(f"  kind     : {kind}")
    print(f"  dt_ctrl  : {dt_ctrl_s}")
    print(f"  horizon N: {N_s}")

    if Q is not None:
        qd = np.diag(Q).astype(float)
        # show a compact slice that’s actually useful
        print(f"  diag(Q)  : {np.array2string(qd, precision=3, suppress_small=True)}")
    if R is not None:
        rd = np.diag(R).astype(float)
        print(f"  diag(R)  : {np.array2string(rd, precision=4, suppress_small=True)}")
    if P is not None:
        P_sym = 0.5*(P + P.T)
        lam_min = float(np.linalg.eigvalsh(P_sym).min())
        print(f"  terminal P: shape {P.shape},  min eig = {lam_min:.3e}")

    if (rpm_min is not None) and (rpm_max is not None):
        print(f"  RPM bounds: [{float(rpm_min):.0f}, {float(rpm_max):.0f}]")
    if (umin is not None) and (umax is not None):
        print(f"  u-box     : min={umin.round(3)}, max={umax.round(3)}")

    # A tiny hint about decision variables when it’s the nonlinear MPC payload pattern
    if kind == "Nonlinear MPC":
        print(f"  decision : W = Ω^2   (u = M W; RPM constraints only)")

def _print_eigs(eigs: np.ndarray, title: str):
    vals = np.asarray(eigs, complex).copy()
    idx  = np.argsort(vals.real)
    vals = vals[idx]
    print(title)
    print("-"*len(title))
    print("  idx            Re(lambda)           Im(lambda)")
    for i, lam in enumerate(vals):
        print(f"{i:5d}   {lam.real:16.8e}   {lam.imag:16.8e}")
    print()

def _detect_sim_kind_keys(keys: set[str]) -> str:
    if "X_inf" in keys or "O_inf" in keys:
        return "sim_lin"
    if "X" in keys or "O" in keys:
        return "sim_nonlin"
    return "unknown"

def view_artifact(path: str):
    print(f"\n=== LQR artifact ===\n{path}")
    _, bag, meta = load_lqr_artifact(ARTIFACTS_DIR, LQR_PREFIX, path)

    A, B = bag["A"], bag["B"]
    Q, R = bag["Q"], bag["R"]
    P, K = bag["P"], bag["K"]
    C    = bag.get("Ctrb", None)

    print(f"shapes: A{A.shape}, B{B.shape}, Q{Q.shape}, R{R.shape}, P{P.shape}, K{K.shape}")
    if C is not None:
        rankC = int(np.linalg.matrix_rank(C))
        print(f"controllability rank: {rankC} / {A.shape[0]}")
    if "eig_ol" in bag: _print_eigs(bag["eig_ol"], "Open-loop eigenvalues (A)")
    if "eig_cl" in bag: _print_eigs(bag["eig_cl"], "Closed-loop eigenvalues (A - B K_inf)")
    P_sym = 0.5*(P + P.T)
    lam_min = float(np.linalg.eigvalsh(P_sym).min())
    print(f"P positive-definite? {lam_min > 0.0}  (min eigenvalue = {lam_min:.3e})")

def view_simulation(path: str):
    print(f"\n=== Simulation ===\n{path}")
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)

    T = data["T"]
    xr = data["xr"]
    kind = _detect_sim_kind_keys(keys)

    rpm_min = float(data.get("rpm_min", 0.0))
    rpm_max = float(data.get("rpm_max", 0.0))


    X = data["X"]; O = data["O"]
    J_line = f"J_nl = {float(data.get('J_nl', np.nan)):.6e}"
    sat_line = f"sat steps = {int(data.get('sat_steps', -1))}/{T.size}"
    print("type: nonlinear")

    print(J_line + "   |   " + sat_line)
    _print_mpc_params(data, keys)

    # Save figures exactly like the sim scripts
    if SAVE:
        apply_style(theme="latex", context="paper", font_scale=1.0, preset="pub")
        params = QuadParams()

        base_dir = os.path.dirname(path)
        stem     = os.path.splitext(os.path.basename(path))[0]

        states_pdf = os.path.join(base_dir, f"{stem}_states.pdf")
        rpms_pdf   = os.path.join(base_dir, f"{stem}_rpms.pdf")
        _plot_states_2x2(T, X, xr, states_pdf, lw=2.0, legend_fs=12, title_fs=14)
        _plot_rpms(T, O, params, rpms_pdf, lw=2.0, legend_fs=12)
        print(f"plots -> {states_pdf}")
        print(f"plots -> {rpms_pdf}")

        # Optional PMP figures (if present in NPZ)
        try:
            if 'lam_dyn' in keys and 'lam_hat' in keys and data['lam_dyn'].size > 0:
                costates_pdf = os.path.join(base_dir, f"{stem}_costates.pdf")
                _plot_costates_stack(T, data['lam_dyn'], data['lam_hat'], costates_pdf, legend_fs=10)
                print(f"plots -> {costates_pdf}")
            if 'ru' in keys and 'rp' in keys and data['ru'].size > 0:
                pmp_pdf = os.path.join(base_dir, f"{stem}_pmp_residuals.pdf")
                _plot_pmp_residuals_inf(T, data['ru'], data['rp'], pmp_pdf)
                print(f"plots -> {pmp_pdf}")
        except Exception:
            pass
    labs = state_labels()
    try:
        idxs = data["metrics_idx"].astype(int)
        tol  = float(np.atleast_1d(data.get("settle_tol", np.array([0.02])))[0])
        if kind == "sim_lin":
            ts  = data.get("ts_inf", None)
            osp = data.get("os_inf_pct", None)
            tag = "linear"
        else:
            ts  = data.get("ts_nl", None)
            osp = data.get("os_nl_pct", None)
            tag = "nonlinear"
        if ts is not None and osp is not None:
            ts  = np.asarray(ts, float)
            osp = np.asarray(osp, float)
            print(f"\nSettling/overshoot ({tag}, {tol*100:.0f}% band):")
            for i, tsi, osi in zip(idxs, ts, osp):
                print(f"  {labs[i]:>6}:  Ts = {tsi:7.3f} s   OS = {osi:6.2f} %")
        # Peak |phi|, |theta| (wrapped), and |psi - psi_ref| (wrapped), in radians
        phi_pk_deg   = float(np.max(np.abs(wrap_pi(X[:, 6]))))
        theta_pk_deg = float(np.max(np.abs(wrap_pi(X[:, 7]))))
        psi_ref      = float(np.asarray(xr)[8])
        psi_pk_deg   = float(np.max(np.abs(wrap_pi(X[:, 8] - psi_ref))))
        print(f"Peak |phi|, |theta|, |psi-psi_ref| [rad] = [{phi_pk_deg:.6f} {theta_pk_deg:.6f} {psi_pk_deg:.6f}]")

    except Exception:
        pass

        # PMP residual metrics (if arrays present)
    try:
        if 'ru' in keys and 'rp' in keys:
            ru = np.asarray(data['ru']); rp = np.asarray(data['rp'])
            if ru.size and rp.size:
                ru_n = np.linalg.norm(ru, ord=np.inf, axis=1)
                rp_n = np.linalg.norm(rp, ord=np.inf, axis=1)
                print("\n[PMP]")
                print(f"  max ||r_u||_inf = {float(np.max(ru_n)):.3e}")
                print(f"  max ||r_p||_inf = {float(np.max(rp_n)):.3e}")
                if 'lam_dyn' in keys and 'lamN' in keys and data['lam_dyn'].size > 0:
                    term_err = float(np.max(np.abs(data['lam_dyn'][-1] - data['lamN'])))
                    print(f"  terminal mismatch = {term_err:.3e}")
    except Exception:
        pass

if __name__ == "__main__":
    apply_style(theme="latex", context="paper", font_scale=1.0)

    any_done = False

    # Loop artifacts
    if ARTIFACT_PATHS:
        for p in ARTIFACT_PATHS:
            if not p: continue
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            view_artifact(p)
            any_done = True

    # Loop simulations
    if SIM_PATHS:
        for p in SIM_PATHS:
            if not p: continue
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            view_simulation(p)
            any_done = True

    if not any_done:
        print("Set ARTIFACT_PATHS and/or SIM_PATHS (lists of paths) at the top of this file.")
