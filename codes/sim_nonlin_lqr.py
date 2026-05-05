#!/usr/bin/env python3
from __future__ import annotations
"""
lqr_nonlin.py
Run steady-state LQR (LQR_inf) on the FULL NONLINEAR quadrotor for ONE setpoint.

- Loads newest LQR artifact if available; else designs LQR_inf at hover.
- Tracks a constant setpoint (px, py, pz, yaw) from macros.SETPOINT.
- Enforces RPM limits each step using the same allocation-based saturation as linear.
- Saves a compact .npz + two clean PDFs:
    1) *_states.pdf : 2×2 panels (pos, vel, angles, rates) + dotted refs
    2) *_rpms.pdf   : Ω_i traces with dashed min/max RPM
"""

import os, numpy as np, matplotlib.pyplot as plt
from datetime import datetime

from macros import (
    ARTIFACTS_DIR, SIM_DIR, DT, T_SINGLE_SETPOINT,
    SETPOINT, Q_matrix, R_matrix, LQR_PREFIX, SIM_PREFIX
)
from plotting import apply_style, save_fig, state_labels, _plot_rpms, _plot_states_2x2, _legend_outside_right
from dynamics import (
    QuadParams, f_dynamics, linearize_hover,
    omegas2_from_thrust_torques, thrust_torques_from_omegas2, clip_omegas
)
from lqr import design_lqr
from utils import rk4_step, save_npz_package, load_lqr_artifact, ct_cost, apply_rpm_saturation, step_metrics_multi

if __name__ == "__main__":
    # Publication preset: bigger ticks/labels/lines
    apply_style(theme="latex", context="paper", font_scale=1.0, preset="pub")
    np.set_printoptions(precision=4, suppress=True)

    params = QuadParams()

    # Try to load an existing LQR artifact; otherwise design fresh
    try:
        art_path, bag, _meta = load_lqr_artifact(ARTIFACTS_DIR, LQR_PREFIX, None)
        A, B = bag["A"], bag["B"]
        x_eq, u_eq = bag["x0"], bag["u0"]
        K, Q, R = bag["K"], bag["Q"], bag["R"]
        src = f"artifact: {art_path}"
    except FileNotFoundError:
        A, B, x_eq, u_eq = linearize_hover(params, yaw=0.0)
        lqr = design_lqr(A, B, Q=Q_matrix(), R=R_matrix(), traj=False)
        K, Q, R = lqr["K"], lqr["Q"], lqr["R"]
        src = "fresh LQR_inf design"

    # Reference & initial condition (deviation zero)
    xr = x_eq.copy()
    xr[0] = float(SETPOINT["px"]); xr[1] = float(SETPOINT["py"]); xr[2] = float(SETPOINT["pz"])
    xr[8] = float(SETPOINT["yaw"])
    x0 = x_eq.copy()

    # Time grid
    dt = float(DT); Tfinal = float(T_SINGLE_SETPOINT)
    N = int(np.floor(Tfinal/dt)) + 1
    T = np.linspace(0.0, Tfinal, N)

    # Allocate
    nx, nu = A.shape[0], B.shape[1]
    X = np.zeros((N, nx)); U_abs = np.zeros((N, nu))  # absolute [T, τx, τy, τz]
    U_dev = np.zeros((N, nu))                         # deviation (for cost/report)
    O = np.zeros((N, 4))                              # motor RPMs
    X[0] = x0

    def f(x, u): return f_dynamics(x, u, params, u_mode="Ttau")

    # Nonlinear rollout with the SAME saturation path as linear
    sat_hits = 0
    for k in range(N-1):
        u_dev_cmd = -K @ (X[k] - xr)   # deviation command
        u_dev_sat, omega = apply_rpm_saturation(
            u_dev_cmd, u_eq, params,
            omegas2_from_thrust_torques, thrust_torques_from_omegas2, clip_omegas
        )
        sat_hits += int(np.any(np.abs(u_dev_cmd - u_dev_sat) > 1e-12))
        U_dev[k] = u_dev_sat
        U_abs[k] = u_eq + u_dev_sat
        O[k] = omega
        X[k+1] = rk4_step(f, X[k], U_abs[k], dt)

    u_dev_last = -K @ (X[-1] - xr)
    U_dev[-1], O[-1] = apply_rpm_saturation(
        u_dev_last, u_eq, params,
        omegas2_from_thrust_torques, thrust_torques_from_omegas2, clip_omegas
    )
    U_abs[-1] = u_eq + U_dev[-1]

    # Cost (continuous-time, deviation variables like linear)
    Xref = np.tile(xr, (N, 1))
    J_nl = ct_cost(T, X, U_dev, Q, R, Xref)

    # step metrics (2%) for px, py, pz, psi
    ref_indices = (0, 1, 2, 6, 7, 8)
    labs = state_labels()
    idxs_nl, ts_nl, os_nl = step_metrics_multi(T, X, xr, indices=ref_indices, tol=0.02)

    print("\nSettling/overshoot (nonlinear, 2% band):")
    for i, tsi, osi in zip(idxs_nl, ts_nl, os_nl):
        print(f"  {labs[i]:>6}:  Ts = {tsi:7.3f} s   OS = {osi:6.2f} %")

    # Report (unified style)
    print("\n=== Nonlinear closed-loop (LQR_inf, single setpoint) ===")
    print(f"{src}")
    print(f"T={Tfinal:.2f}s, dt={dt:.4f}s, steps={N}")
    print(f"J (LQR_inf) = {J_nl:.6e}   | saturation steps = {sat_hits}/{N}")

    # Save (overwrite each run, like linear)
    os.makedirs(SIM_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); stamp = ''  # overwrite pattern
    base_inf = os.path.join(SIM_DIR, f"{SIM_PREFIX['INF']}{'nonlin_'}{stamp}")
    base_    = os.path.join(SIM_DIR, f"{'lqr_nonlin_'}{stamp}")
    npz_path = base_ + ".npz"

    payload = {
        "T": T,
        "Q": Q, "R": R,
        "xr": xr,
        "x0": x0,
        "u_eq": u_eq,
        "X": X,
        "U_abs": U_abs,
        "U_dev": U_dev,
        "O": O,
        "J_nl": float(J_nl),
        "sat_steps": int(sat_hits),
        "rpm_min": float(params.min_rpm),
        "rpm_max": float(params.max_rpm),
        # step metrics (nonlinear)
        "metrics_idx": idxs_nl.astype(np.int32),
        "settle_tol": np.array([0.02], dtype=float),
        "ts_nl": ts_nl.astype(float),
        "os_nl_pct": os_nl.astype(float),
    }
    save_npz_package(npz_path, payload, {"desc": "Nonlinear closed-loop LQR_inf (with RPM saturation)"})
    print(f"saved -> {npz_path}")

    # Publication-quality plots (2×2 states + RPMs)
    states_pdf = base_inf + "_states.pdf"
    rpms_pdf   = base_inf + "_rpms.pdf"
    _plot_states_2x2(T, X, xr, states_pdf, lw=2.0, legend_fs=12, title_fs=14)
    _plot_rpms(T, O, params, rpms_pdf, lw=2.0, legend_fs=12)
    print(f"plots -> {states_pdf}")
    print(f"plots -> {rpms_pdf}")

    print("done.")
