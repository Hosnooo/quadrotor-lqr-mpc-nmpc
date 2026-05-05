#!/usr/bin/env python3
from __future__ import annotations
"""
sim_mpc_linear.py
Closed-loop simulation of the nonlinear quadrotor using a **linear MPC** controller (deviation form).

"""

import os, time, numpy as np, matplotlib.pyplot as plt
from datetime import datetime

from macros import DT, SETPOINT, Q_matrix, R_matrix, SIM_DIR, SIM_PREFIX, T_SINGLE_SETPOINT
from utils import c2d_series, rk4_step, save_npz_package, apply_rpm_saturation, wrap_pi
from utils import ct_cost, step_metrics_multi
from plotting import apply_style, _plot_states_2x2, _plot_rpms
from dynamics import (
    QuadParams, linearize_hover, f_dynamics,
    omegas2_from_thrust_torques, thrust_torques_from_omegas2, clip_omegas
)
from mpc_linear import LinearMPC, LinearMPCConfig, dare_terminal_weight, box_bounds_from_rpm

def _xref_from_setpoint(x_eq: np.ndarray) -> np.ndarray:
    """Reference state vector (12,) from SETPOINT: px,py,pz,yaw."""
    xr = np.zeros_like(x_eq)
    xr[0] = float(SETPOINT.get("px", 0.0))
    xr[1] = float(SETPOINT.get("py", 0.0))
    xr[2] = float(SETPOINT.get("pz", 0.0))
    xr[8] = float(SETPOINT.get("yaw", 0.0))
    return xr

def main():
    # Params and linear model at hover
    params = QuadParams()
    A, B, x_eq, u_eq = linearize_hover(params, yaw=0.0)

    # MPC rate (snapped to integer multiple of DT)
    dt_plant = float(DT)
    dt_mpc_des = 0.01   # 100 Hz
    k_mpc = max(1, int(round(dt_mpc_des / dt_plant)))
    dt_mpc = k_mpc * dt_plant
    Ad, Bd = c2d_series(A, B, dt_mpc, order=10)

    # Weights and terminal
    Q = Q_matrix(); R = R_matrix()
    P = dare_terminal_weight(Ad, Bd, Q, R)

    # Horizon and input bounds from rpm box
    N = 30
    umin, umax = box_bounds_from_rpm(
        params, omegas2_from_thrust_torques, thrust_torques_from_omegas2,
        params.min_rpm, params.max_rpm
    )
    pad = 1e-6; umin = umin + pad; umax = umax - pad

    # Build MPC (deviation form)
    cfg = LinearMPCConfig(Ad=Ad, Bd=Bd, Q=Q, R=R, P=P, N=N, umin=umin, umax=umax)
    mpc = LinearMPC(cfg)
    # mpc.solve_kwargs["verbose"] = True  # uncomment for OSQP logs

    # References
    xr = _xref_from_setpoint(x_eq)
    ur = u_eq.copy()

    # Initial state (store x0 for payload)
    x0 = x_eq.copy()
    x = x0.copy()
    # x[:3] += np.array([0.25, -0.25, 0.0])

    # Time grid
    Tfinal = float(T_SINGLE_SETPOINT)
    N_steps = int(np.floor(Tfinal / dt_plant)) + 1
    T = np.linspace(0.0, (N_steps - 1) * dt_plant, N_steps)

    # Buffers
    nx, nu = A.shape[0], B.shape[1]
    X = np.zeros((N_steps, nx))
    U_cmd = np.zeros((N_steps, nu))     # absolute command from MPC (pre-sat)
    U = np.zeros((N_steps, nu))         # absolute applied (post-sat)
    O = np.zeros((N_steps, 4))          # RPM

    # -------- console header (meaningful stuff) ----------
    EPS = 1e-6
    print("[MPC] config")
    print(f"  dt_plant={dt_plant:.4f}s  dt_mpc={dt_mpc:.4f}s  horizon N={N}  steps={N_steps}  T={T[-1]:.2f}s")
    print(f"  Qdiag[:6]={np.diag(Q)[:6].round(3)}  Rdiag={np.diag(R).round(4)}")
    print(f"  u_eq  = {ur.round(5)}")
    print(f"  u box = [{umin.round(3)}] .. [{umax.round(3)}]")
    pos_off = (x[:3] - xr[:3]).round(3)
    yaw_off = float(np.degrees(wrap_pi(x[8] - xr[8])))
    print(f"  x0 offsets: pos={pos_off} m, yaw={yaw_off:+.2f}°")

    # Progress cadence
    progress_every = 0.5  # seconds between prints
    _step_interval = max(1, int(round(progress_every / dt_plant)))
    _last_print = -1
    _t0 = time.time()

    # Loop
    apply_style(preset="pub")
    u_last = ur.copy()
    sat_hits = 0  # count how many steps hit saturation (any rotor)
    for k in range(N_steps):
        # Receding-horizon MPC (absolute u)
        if k % k_mpc == 0:
            u0 = mpc.make_step(x, xr, ur, u_prev=u_last)
            u_last = u0.copy()
        else:
            u0 = u_last
        U_cmd[k] = u0

        # Saturate in RPM space (deviation-input API)
        u_dev_cmd = u0 - ur
        u_dev_feas, omega = apply_rpm_saturation(
            u_dev_cmd, u_eq=ur, params=params,
            f_u2w2=omegas2_from_thrust_torques,
            f_w22u=thrust_torques_from_omegas2,
            clip_fun=clip_omegas
        )
        # Count saturation if the feasible dev input differs
        if np.any(np.abs(u_dev_cmd - u_dev_feas) > 1e-12):
            sat_hits += 1

        u_sat = ur + u_dev_feas
        O[k] = omega
        U[k] = u_sat

        # Nonlinear plant step
        x = rk4_step(lambda xx, uu: f_dynamics(xx, uu, params, u_mode="Ttau"), x, u_sat, dt_plant)
        X[k] = x

        # -------- informative progress line ----------
        ep = x[:3] - xr[:3]
        ep_norm = float(np.linalg.norm(ep))
        ez = float(ep[2])
        eyaw_deg = float(np.degrees(wrap_pi(x[8] - xr[8])))
        rot_sat = int(((omega <= params.min_rpm + EPS) | (omega >= params.max_rpm - EPS)).sum())
        u_bound_hits = int((np.isclose(u_sat, umax, atol=1e-7) | np.isclose(u_sat, umin, atol=1e-7)).sum())

        if (k == N_steps - 1) or (k - _last_print) >= _step_interval:
            pct = 100.0 * (k + 1) / N_steps
            print(
                f"\r[MPC] {pct:5.1f}%  t={T[k]:.2f}/{T[-1]:.2f}s"
                f" | ||e_p||={ep_norm:.3f} m, e_z={ez:+.3f} m, e_yaw={eyaw_deg:+.1f}°"
                f" | rot_sat={rot_sat}/4  u@bounds={u_bound_hits}/4",
                end="", flush=True
            )
            _last_print = k
    print()

    # Metrics
    ref_indices = (0, 1, 2, 6, 7, 8)
    idxs_nl, ts_nl, os_nl = step_metrics_multi(T, X, xr, indices=ref_indices, tol=0.02)
    J_nl = ct_cost(T, X, U - u_eq, Q, R, xr)

    # -------- end-of-run summary ----------
    pos_final = X[-1, :3] - xr[:3]
    yaw_final_deg = float(np.degrees(wrap_pi(X[-1, 8] - xr[8])))
    att_max_deg = np.degrees(np.max(np.abs(X[:, 6:9]), axis=0))  # |phi|,|theta|,|psi| max
    rot_any_sat_steps = int(np.count_nonzero(
        (O <= params.min_rpm + EPS) | (O >= params.max_rpm - EPS)
    ))
    sat_pct = 100.0 * rot_any_sat_steps / N_steps

    print("[MPC] summary")
    print(f"  J_nl={J_nl:.4g}  (continuous-time cost)")
    print(f"  final pos err = {pos_final.round(3)} m   final yaw err = {yaw_final_deg:+.2f}°")
    print(f"  max |phi,theta,psi| = {att_max_deg.round(2)} °")
    print(f"  RPM range = [{O.min():.0f}, {O.max():.0f}]   steps with any rotor saturated = {rot_any_sat_steps}/{N_steps} ({sat_pct:.1f}%)")
    names = ["px","py","pz","phi","theta","psi"]
    for i, nm in enumerate(names):
        print(f"  {nm:>5}: ts={ts_nl[i]:.2f}s  overshoot={os_nl[i]:.1f}%")

    # Payload (mirror keys/types you use elsewhere)
    payload = {
        "T": T,
        "Q": Q, "R": R, "P": P,
        "xr": xr,
        "x0": x0,
        "u_eq": u_eq,
        "X": X,
        "U_abs": U,
        "U_dev": U - u_eq,
        "O": O,
        "J_nl": float(J_nl),
        "sat_steps": int(sat_hits),
        "rpm_min": float(params.min_rpm),
        "rpm_max": float(params.max_rpm),
        "metrics_idx": idxs_nl.astype(np.int32),
        "settle_tol": np.array([0.02], dtype=float),
        "ts_nl": ts_nl.astype(float),
        "os_nl_pct": os_nl.astype(float),
        "A": A, "B": B, "Ad": Ad, "Bd": Bd,

        # >>> NEW: viewer-friendly MPC metadata
        "dt_ctrl": float(dt_mpc),
        "N_horizon": int(N),
        "umin": umin.astype(float),
        "umax": umax.astype(float),
    }

    # Base path + save (exact print style)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); stamp = ''  # same overwrite pattern you use
    base  = os.path.join(SIM_DIR, f"{'mpc_lin_'}{stamp}")
    npz_path = base + ".npz"
    save_npz_package(npz_path, payload, {"desc": "Linear MPC closed-loop (nonlinear plant) with RPM saturation"})
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
