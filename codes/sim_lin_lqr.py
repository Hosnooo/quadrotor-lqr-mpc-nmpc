#!/usr/bin/env python3
from __future__ import annotations
"""
sim_lin_lqr.py
Linear closed-loop simulations using an existing LQR artifact.

- Loads newest ARTIFACTS_DIR/LQR_PREFIX*.npz.
- Simulates LQR_\infty and, if schedules exist, TV-LQR over [0, T].
- Reference: constant step on (p_x, p_y, p_z, psi) using macros.SETPOINT.
- Applies RPM saturation each step using dynamics allocation utilities.
- Saves one .npz with trajectories (including RPMs) and separate PDFs for each controller.
- Plots: 2×2 state panels + RPMs with dashed saturation lines, publication-ready style.
"""

import os as _os, numpy as np, matplotlib.pyplot as plt
from datetime import datetime

from macros import (
    ARTIFACTS_DIR, SIM_DIR, DT, T_SINGLE_SETPOINT,
    SETPOINT, newest_path, LQR_PREFIX, SIM_PREFIX
)
from plotting import apply_style, save_fig, state_labels, _plot_states_2x2, _legend_outside_right, _plot_rpms
from utils import save_npz_package, load_lqr_artifact, rk4_step_lin, apply_rpm_saturation, ct_cost, step_metrics_multi,_scalar_ref, peak_abs_deflection
from lqr import finite_horizon_pmp
from dynamics import (
    QuadParams,
    omegas2_from_thrust_torques,
    thrust_torques_from_omegas2,
    clip_omegas,
)

if __name__ == "__main__":
    # Global publication style: bigger ticks/labels/lines everywhere
    apply_style(theme="latex", context="paper", font_scale=1.0, preset="pub")

    # Load newest artifact
    art_path, bag, meta = load_lqr_artifact(ARTIFACTS_DIR, LQR_PREFIX, None)
    A, B   = bag["A"], bag["B"]
    x_eq   = bag["x0"]; u_eq = bag["u0"]
    Q, R   = bag["Q"], bag["R"]
    K      = bag["K"]
    has_tv = all(k in bag for k in ("t_fh", "P_seq", "K_seq"))

    # Params and reference
    params = QuadParams()
    xr = np.zeros_like(x_eq)
    xr[0] = float(SETPOINT["px"]); xr[1] = float(SETPOINT["py"]); xr[2] = float(SETPOINT["pz"])
    xr[8] = float(SETPOINT["yaw"])

    # Time grid
    Tfinal = float(T_SINGLE_SETPOINT)
    dt     = float(DT)
    N = int(np.floor(Tfinal/dt)) + 1
    T = np.linspace(0.0, Tfinal, N)

    # Allocate arrays
    nx, nu = B.shape
    X = np.zeros((N, nx)); U = np.zeros((N, nu))
    O = np.zeros((N, 4))
    X[0] = np.zeros(nx)

    # LQR_\infty rollout with saturation
    sat_hits = 0
    for k in range(N-1):
        u_dev = -K @ (X[k] - xr)
        u_dev_sat, omega = apply_rpm_saturation(
            u_dev, u_eq, params,
            omegas2_from_thrust_torques, thrust_torques_from_omegas2, clip_omegas
        )
        sat_hits += int(np.any(np.abs(u_dev - u_dev_sat) > 1e-12))
        U[k] = u_dev_sat
        O[k] = omega
        X[k+1] = rk4_step_lin(A, B, X[k], u_dev_sat, dt)
    u_last = -K @ (X[-1] - xr)
    U[-1], O[-1] = apply_rpm_saturation(
        u_last, u_eq, params,
        omegas2_from_thrust_torques, thrust_torques_from_omegas2, clip_omegas
    )

    # Cost for LQR_\infty
    Xref = np.tile(xr, (N,1))
    J = ct_cost(T, X, U, Q, R, Xref)

    ref_indices = (0, 1, 2, 6, 7, 8)
    labs = state_labels()
    idxs, ts, os = step_metrics_multi(T, X, xr, indices=ref_indices, tol=0.02)

    print("\nSettling/overshoot (linear, 2% band):")
    for i, tsi, osi in zip(idxs, ts, os):
        print(f"  {labs[i]:>6}:  Ts = {tsi:7.3f} s   OS = {osi:6.2f} %")

    # Report
    print("\n=== Linear closed-loop (using LQR artifact) ===")
    print(f"artifact: {art_path}")
    print(f"T={Tfinal:.2f}s, dt={dt:.4f}s, steps={N}")
    print(f"J (LQR) = {J:.6e}   | saturation steps = {sat_hits}/{N}")

    # Save trajectories (overwrite each run)
    _os.makedirs(SIM_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); stamp = ''
    base = _os.path.join(SIM_DIR, f"{SIM_PREFIX['INF']}{'lin_'}{stamp}")

    payload = {
        "T": T,
        "Q": Q, "R": R,
        "xr": xr,
        "u_eq": u_eq,
        "X": X,
        "U": U,
        "O": O,
        "J": float(J),
        "sat_steps": int(sat_hits),
        "rpm_min": float(params.min_rpm),
        "rpm_max": float(params.max_rpm),
        # step metrics (linear)
        "metrics_idx": idxs.astype(np.int32),
        "settle_tol": np.array([0.02], dtype=float),
        "ts": ts.astype(float),
        "os_pct": os.astype(float),
    }

    base_ = _os.path.join(SIM_DIR, f"{'lqr_lin_'}{stamp}")
    npz_path = base_ + ".npz"
    save_npz_package(npz_path, payload, {"desc": "Linear closed-loop LQR sims (with RPM saturation)"})
    print(f"saved -> {npz_path}")

    # Publication-quality plots
    # LQR_\infty
    states_pdf = base + "_states.pdf"
    rpms_pdf   = base + "_rpms.pdf"
    _plot_states_2x2(T, X, xr, states_pdf, lw=2.0, legend_fs=12, title_fs=14)
    _plot_rpms(T, O, params, rpms_pdf, lw=2.0, legend_fs=12)
    print(f"plots -> {states_pdf}")
    print(f"plots -> {rpms_pdf}")

    print("done.")
