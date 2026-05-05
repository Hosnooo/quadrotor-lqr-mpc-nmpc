#!/usr/bin/env python3
from __future__ import annotations
"""
sim_mpc_nonlinear.py
Closed-loop simulation of the nonlinear quadrotor using **nonlinear MPC**
(multiple shooting, RK4 via casadi_ms).
"""

import os, numpy as np, matplotlib.pyplot as plt
from datetime import datetime

from macros import DT, SETPOINT, Q_matrix, R_matrix, SIM_DIR, SIM_PREFIX, T_SINGLE_SETPOINT
from plotting import apply_style, _plot_states_2x2, _plot_rpms, state_labels
from utils import rk4_step, save_npz_package, ct_cost, step_metrics_multi, wrap_pi, apply_rpm_saturation
from dynamics import (
    QuadParams, linearize_hover, f_dynamics,
    omegas2_from_thrust_torques, thrust_torques_from_omegas2, clip_omegas
)
from mpc_linear import box_bounds_from_rpm
from mpc_nonlinear import NLMPCConfig, NonlinearMPC_MS

def _xref_from_setpoint(x_eq: np.ndarray) -> np.ndarray:
    xr = np.array(x_eq, copy=True)
    xr[0] = float(SETPOINT.get("px", 0.0))
    xr[1] = float(SETPOINT.get("py", 0.0))
    xr[2] = float(SETPOINT.get("pz", 0.0))
    xr[8] = float(SETPOINT.get("yaw", 0.0))
    xr[3:6] = 0.0
    xr[6:8] = 0.0
    xr[9:12] = 0.0
    return xr

def main():
    apply_style(preset="pub")
    np.set_printoptions(precision=4, suppress=True)

    # Params & hover
    params = QuadParams()
    A, B, x_eq, u_eq = linearize_hover(params, yaw=0.0)

    # Timing
    dt = float(DT)
    Tf = float(T_SINGLE_SETPOINT)
    N_steps = int(round(Tf / dt)) + 1
    dt_mpc_des = 0.02
    k_mpc = max(1, int(round(dt_mpc_des / dt)))
    dt_ctrl = k_mpc * dt

    # Horizon/weights
    N = 80
    Q, R = Q_matrix(), R_matrix()
    Qf = 10.0 * Q.copy()

    # Report u-box (for parity with linear sim)
    umin, umax = box_bounds_from_rpm(
        params, omegas2_from_thrust_torques, thrust_torques_from_omegas2,
        params.min_rpm, params.max_rpm
    )
    pad = 1e-6; umin = umin + pad; umax = umax - pad

    print("=== Nonlinear MPC (multiple-shooting RK4) ===")
    print(f"T={Tf:.2f}s, dt={dt:.4f}s, steps={N_steps}, horizon={N}, dt_ctrl={dt_ctrl:.4f}s")
    print(f"u box (reporting): min={umin.round(4)}, max={umax.round(4)}")

    # Controller (decision = W2 with RPM bounds)
    cfg = NLMPCConfig(
        N=N, dt=dt_ctrl, Q=Q, R=R, Qf=Qf,
        rpm_min=params.min_rpm, rpm_max=params.max_rpm,
        du_max=None, ipopt_max_iter=600, ipopt_tol=1e-4, verbose=False
    )
    nmpc = NonlinearMPC_MS(params, cfg)

    # References
    xr = _xref_from_setpoint(x_eq)
    ur = u_eq.copy()

    # Initial condition (use SETPOINT offsets if provided)
    x = x_eq.copy()
    x[0] += float(SETPOINT.get("px0",  -1.0))
    x[1] += float(SETPOINT.get("py0",   1.0))
    x[2] += float(SETPOINT.get("pz0",   3.0))
    x[8] += float(SETPOINT.get("yaw0",  np.radians(-3.0)))

    pos_off = (x[:3] - xr[:3]).round(3)
    yaw_off = float(np.degrees(wrap_pi(x[8] - xr[8])))
    print(f"  x0 offsets: pos={pos_off} m, yaw={yaw_off:+.2f}°")

    # Buffers
    T = np.linspace(0.0, Tf, N_steps)
    X = np.zeros((N_steps, x.size)); X[0] = x
    U_cmd = np.zeros((N_steps, 4))
    U = np.zeros((N_steps, 4))
    O = np.zeros((N_steps, 4))

    # Warm-starts
    Xw = np.tile(x.reshape(-1,1), (1, N+1))
    W2w = np.tile((params.min_rpm**2) * np.ones(4), (N, 1)).T

    # Previous applied (for Δu if enabled)
    u_prev = ur.copy()
    w2_prev = omegas2_from_thrust_torques(u_prev, params)

    # Progress cadence
    progress_every = 0.5
    step_pr = max(1, int(round(progress_every / dt)))
    last_print = -1

    sat_steps = 0
    for k in range(N_steps):
        # Solve at control ticks
        if k % k_mpc == 0:
            XR = np.tile(xr.reshape(-1,1), (1, N+1))
            UR = np.tile(ur.reshape(-1,1), (1, N))
            nmpc.set_params(x0=x, xr_seq=XR, ur_seq=UR, w2_prev=w2_prev)
            sol = nmpc.solve(warm={"X": Xw, "W2": W2w})
            if sol["status"] != "solved":
                raise RuntimeError(f"NMPC failed: {sol['status']}")
            u0 = sol["U"][:, 0].reshape(-1)
            w2_0 = sol["W2"][:, 0].reshape(-1)
            Xw, W2w = sol["X"], sol["W2"]
            u_prev, w2_prev = u0.copy(), w2_0.copy()
        else:
            u0, w2_0 = u_prev, w2_prev

        U_cmd[k] = u0

        # Apply actuator saturation in RPM space
        u_dev_cmd = u0 - ur
        u_dev_feas, omega = apply_rpm_saturation(
            u_dev_cmd, u_eq=ur, params=params,
            f_u2w2=omegas2_from_thrust_torques,
            f_w22u=thrust_torques_from_omegas2,
            clip_fun=clip_omegas
        )
        u_applied = ur + u_dev_feas

        # Count saturation hits
        rot_sat = int(np.any(np.isclose(omega, params.min_rpm)) or np.any(np.isclose(omega, params.max_rpm)))
        sat_steps += rot_sat

        # Nonlinear plant integrate (Tτ mode)
        x = rk4_step(lambda xx, uu: f_dynamics(xx, uu, params, u_mode="Ttau"), x, u_applied, dt)

        # Log
        X[k] = x
        U[k] = u_applied
        O[k] = omega

        if (k - last_print) >= step_pr or k == N_steps - 1:
            ep = float(np.linalg.norm(x[:3] - xr[:3]))
            ez = float(x[2] - xr[2])
            eyaw = float(np.degrees(wrap_pi(x[8] - xr[8])))
            pct = 100.0 * k / (N_steps - 1)
            print(f"\r[NMPC] {pct:5.1f}%  t={T[k]:.2f}/{T[-1]:.2f}s | ||e_p||={ep:.3f} m, e_z={ez:+.3f} m, e_yaw={eyaw:+.1f}° | rot_sat={rot_sat}/4",
                  end="", flush=True)
            last_print = k
    print()

    # Metrics & cost
    ref_idxs = (0,1,2,6,7,8)
    ts_idx, ts_sec, os_pct = step_metrics_multi(T, X, xr, indices=ref_idxs, tol=0.02)
    J = ct_cost(T, X, U - u_eq, Q, R, xr)

    # Summary
    labs = state_labels()
    print("\nSettling/overshoot (nonlinear, 2% band):")
    for i, tsi, osi in zip(ts_idx, ts_sec, os_pct):
        print(f"  {labs[int(i)]:>6}:  Ts = {float(tsi):7.3f} s   OS = {float(osi):6.2f} %")
    phi_pk   = float(np.max(np.abs(wrap_pi(X[:, 6]))))
    theta_pk = float(np.max(np.abs(wrap_pi(X[:, 7]))))
    psi_pk   = float(np.max(np.abs(wrap_pi(X[:, 8] - xr[8]))))

    # Save NPZ
    payload = {
        "desc": "Nonlinear MPC closed-loop (nonlinear plant) with RPM saturation",
        "DT": dt, "Tf": Tf, "dt_ctrl": dt_ctrl, "N_horizon": int(N),
        "x_eq": x_eq, "u_eq": u_eq, "xr": xr, "ur": ur,
        "T": T, "X": X, "U_cmd": U_cmd, "U_abs": U, "U_dev": U - u_eq,
        "O": O, "J_nl": float(J), "sat_steps": int(sat_steps),
        "rpm_min": float(params.min_rpm), "rpm_max": float(params.max_rpm),

        # >>> NEW: save the input box reported above
        "umin": umin.astype(float),
        "umax": umax.astype(float),

        "metrics_idx": np.asarray(ref_idxs, dtype=np.int32),
        "settle_tol": np.array([0.02], dtype=float),
        "ts_nl": ts_sec.astype(float), "os_nl_pct": os_pct.astype(float),
        "ts_idx": ts_idx.astype(int), "ts_sec": ts_sec.astype(float), "os_pct": os_pct.astype(float),
        "A": A, "B": B,
        "Q": Q, "R": R,
    }

    stamp = ''
    base = os.path.join(SIM_DIR, f"{'mpc_nonlin_'}{stamp}")
    npz_path = base + ".npz"
    save_npz_package(npz_path, payload, {"desc": payload["desc"]})
    print(f"saved -> {npz_path}")

    # Figures
    states_pdf = base + "_states.pdf"
    _plot_states_2x2(T, X, xr, states_pdf, lw=2.0, legend_fs=12, title_fs=14)
    print(f"plots -> {states_pdf}")

    rpms_pdf = base + "_rpms.pdf"
    _plot_rpms(T, O, params, rpms_pdf, lw=2.0, legend_fs=12)
    print(f"plots -> {rpms_pdf}")

if __name__ == "__main__":
    main()
