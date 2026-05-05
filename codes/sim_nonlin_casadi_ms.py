# sim_nonlin_casadi_ms.py
from __future__ import annotations
import os, numpy as np
import matplotlib.pyplot as plt

from macros import SIM_DIR, DT, T_SINGLE_SETPOINT, SETPOINT, Q_matrix, R_matrix, SIM_PREFIX
from plotting import apply_style, save_fig, state_labels, _plot_rpms, _plot_states_2x2, _legend_outside_right, _plot_costates_stack, _plot_pmp_residuals_inf
from dynamics import QuadParams, linearize_hover, omegas2_from_thrust_torques
from utils import ct_cost, save_npz_package, step_metrics_multi
from casadi_ms import ocp_casadi_ms_ipopt

if __name__ == "__main__":
    apply_style(theme="latex", context="paper", font_scale=1.0, preset="pub")
    np.set_printoptions(precision=4, suppress=True)

    # Params & hover
    params = QuadParams()
    A_hover, B_hover, x_eq, u_eq = linearize_hover(params, yaw=0.0)

    # Setpoint refs
    xr = x_eq.copy()
    xr[0] = float(SETPOINT["px"]); xr[1] = float(SETPOINT["py"]); xr[2] = float(SETPOINT["pz"])
    xr[8] = float(SETPOINT["yaw"])
    x0 = x_eq.copy()

    # Weights & horizon (Qf = 10*Q like you had)
    Q  = Q_matrix()
    R  = R_matrix()
    Qf = 10.0 * Q
    Tf = float(T_SINGLE_SETPOINT)

    # Grid (coarse for speed; refine later with warm start)
    N_target = 200
    dt_opt = max(float(DT), Tf / float(N_target))

    x_ref = lambda t: xr
    u_ref = lambda t: u_eq

    # Warm start W2 from hover
    W2_eq = omegas2_from_thrust_torques(u_eq, params)
    N = int(round(Tf / dt_opt)) + 1
    W2_init = np.tile(W2_eq, (N-1, 1))

    print(f"CasADi Multiple Shooting (W2 bounds): Tf={Tf:.3f}s, dt_opt={dt_opt:.4f}s, N={N}", flush=True)

    T, X, U, O, info = ocp_casadi_ms_ipopt(
        params=params, x0=x0, t0=0.0, tf=Tf, dt=dt_opt,
        Q=Q, R=R, Qf=Qf, x_ref=x_ref, u_ref=u_ref,
        W2_init=W2_init, ipopt_max_iter=600, ipopt_verb=5
    )

    # Cost & diagnostics (use applied U and O)
    J_run = ct_cost(T, X, U - u_eq, Q, R, np.tile(xr, (len(T), 1)))
    J_term = 0.5 * ((X[-1] - xr) @ (Qf @ (X[-1] - xr)))
    J_total = float(J_run + J_term)

    print("\n=== Direct multiple shooting (CasADi + IPOPT, hard RPM bounds) ===")
    print(f"Iterations={info.get('iter_count', 'NA')}, success={info.get('success', 'NA')}")
    print(f"J_run = {J_run:.6e}   J_term = {J_term:.6e}   J_total = {J_total:.6e}")
    print(f"gap ||x(T)-xr|| = {np.linalg.norm(X[-1]-xr):.3e}")

    # Settling/overshoot (2% band)
    idxs, Ts, OS = step_metrics_multi(T, X, xr, indices=(0,1,2,6,7,8), tol=0.02)
    labs = state_labels()
    print("\nSettling/overshoot (2% band):")
    for i, tsi, osi in zip(idxs, Ts, OS):
        print(f"  {labs[i]:>6}:  Ts = {tsi:7.3f} s   OS = {osi:6.2f} %")

    # PMP metrics
    m = info.get("pmp_metrics", {})
    print("\n[PMP checks]")
    print(f"max ||r_u||_inf = {m.get('ru_max_inf', 0.0):.3e}")
    print(f"max ||r_p||_inf = {m.get('rp_max_inf', 0.0):.3e}")
    print(f"terminal mismatch = {m.get('term_err_inf', 0.0):.3e}")

    # Save NPZ
    os.makedirs(SIM_DIR, exist_ok=True)
    base = os.path.join(SIM_DIR, f"{SIM_PREFIX.get('OCP','ocp_')}nonlin_ms_casadi_")

    rpm_min = float(params.min_rpm)
    rpm_max = float(params.max_rpm)
    eps = 1e-6
    sat_steps = int(((O <= rpm_min + eps) | (O >= rpm_max - eps)).any(axis=1).sum())
    J_nl = float(J_total)
    ts_nl = Ts.astype(float)
    os_nl_pct = OS.astype(float)

    payload = {
        # trajectories
        "T": T, "X": X,
        "U_abs": U, "U_dev": U - u_eq,
        "O": O, "xr": xr, "u_eq": u_eq,
        # metrics
        "settle_tol": np.array([0.02], dtype=float),
        "metrics_idx": idxs.astype(np.int32), "ts": Ts.astype(float), "os_pct": OS.astype(float),
        "J_nl": J_nl,
        "sat_steps": sat_steps,
        "rpm_min": rpm_min,
        "rpm_max": rpm_max,
        "ts_nl": ts_nl,
        "os_nl_pct": os_nl_pct,
        # PMP artifacts
        "lam_dyn": info.get("lam_dyn", None),
        "lam_hat": info.get("lam_hat", None),
        "ru": info.get("ru", None),
        "rp": info.get("rp", None),
        "lamN": info.get("lamN", None),
        "pmp_metrics": info.get("pmp_metrics", {}),
        "dt": float(dt_opt),
    }

    npz_path = base + ".npz"
    save_npz_package(npz_path, payload, {"desc": "CasADi multiple shooting with hard RPM bounds (W2 vars) + PMP checks"})
    print(f"saved -> {npz_path}")

    # Plots
    states_pdf = base + "_states.pdf"
    _plot_states_2x2(T, X, xr, states_pdf, lw=2.0, legend_fs=12, title_fs=14)
    print(f"plots -> {states_pdf}")

    rpms_pdf = base + "_rpms.pdf"
    _plot_rpms(T, O, params, rpms_pdf, lw=2.0, legend_fs=12)
    print(f"plots -> {rpms_pdf}")

    # New: costates & residuals
    if info.get("lam_dyn", None) is not None:
        lam_pdf = base + "_costates.pdf"
        _plot_costates_stack(T, info["lam_dyn"], info["lam_hat"], lam_pdf, legend_fs=10)
        print(f"plots -> {lam_pdf}")
    if info.get("ru", None) is not None and info.get("rp", None) is not None:
        res_pdf = base + "_pmp_residuals.pdf"
        _plot_pmp_residuals_inf(T, info["ru"], info["rp"], res_pdf)
        print(f"plots -> {res_pdf}")

    print("done.")
