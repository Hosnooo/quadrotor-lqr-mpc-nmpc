#!/usr/bin/env python3
from __future__ import annotations
import os, numpy as np
from macros import DT, SETPOINT, Q_matrix, R_matrix, SIM_DIR, T_SINGLE_SETPOINT
from plotting import apply_style
from utils import rk4_step, save_npz_package, ct_cost, step_metrics_multi, wrap_pi, apply_rpm_saturation
from dynamics import QuadParams, linearize_hover, f_dynamics, \
    omegas2_from_thrust_torques, thrust_torques_from_omegas2, clip_omegas
from mpc_nonlinear import NLMPCConfig, NonlinearMPC_MS

DT_CTRL = 0.02
N_LIST  = [30, 40, 60, 80, 100]
RESULTS_SUBDIR = "NMPC_sweepN_dt020"

def _xref(x_eq):
    xr = np.array(x_eq, copy=True)
    xr[0] = float(SETPOINT.get("px", 0.0))
    xr[1] = float(SETPOINT.get("py", 0.0))
    xr[2] = float(SETPOINT.get("pz", 0.0))
    xr[8] = float(SETPOINT.get("yaw", 0.0))
    xr[3:6] = 0.0; xr[6:8] = 0.0; xr[9:12] = 0.0
    return xr

def _x0(x_eq):
    x = np.array(x_eq, copy=True)
    x[0] += float(SETPOINT.get("px0", -1.0))
    x[1] += float(SETPOINT.get("py0",  1.0))
    x[2] += float(SETPOINT.get("pz0",  3.0))
    x[8] += float(SETPOINT.get("yaw0", np.radians(-3.0)))
    return x

def run_one(N:int, params:QuadParams, dt:float, dt_ctrl:float, x0, xr, ur):
    Q,R = Q_matrix(), R_matrix()
    Qf  = 10.0*Q.copy()
    cfg = NLMPCConfig(N=N, dt=dt_ctrl, Q=Q, R=R, Qf=Qf,
                      rpm_min=float(params.min_rpm), rpm_max=float(params.max_rpm),
                      du_max=None, ipopt_max_iter=600, ipopt_tol=1e-4, verbose=False)
    nmpc = NonlinearMPC_MS(params, cfg)

    Tf = float(T_SINGLE_SETPOINT)
    n_steps = int(round(Tf/dt)) + 1
    T = np.linspace(0.0, (n_steps-1)*dt, n_steps)
    k_mpc = max(1, int(round(dt_ctrl/dt)))

    X = np.zeros((n_steps,12)); U = np.zeros((n_steps,4)); O = np.zeros((n_steps,4))
    x = np.array(x0, copy=True)
    Xw = np.tile(x.reshape(-1,1), (1, N+1))
    W2w = np.tile((params.min_rpm**2)*np.ones(4), (N,1)).T
    u_prev = ur.copy()
    w2_prev = omegas2_from_thrust_torques(u_prev, params)
    sat_steps = 0
    print(f"[NMPC N={N:3d}] start (dt_ctrl={dt_ctrl:.3f}s, steps={n_steps})")
    for k in range(n_steps):
        if k % k_mpc == 0:
            XR = np.tile(xr.reshape(-1,1), (1, N+1))
            UR = np.tile(ur.reshape(-1,1), (1, N))
            nmpc.set_params(x0=x, xr_seq=XR, ur_seq=UR, w2_prev=w2_prev)
            sol = nmpc.solve(warm={"X":Xw, "W2":W2w})
            if sol["status"] != "solved":
                return {"status":f"failed:{sol['status']}", "N":N, "T":T[:k], "X":X[:k], "U":U[:k], "O":O[:k]}
            u0 = sol["U"][:,0]; w2_0 = sol["W2"][:,0]
            Xw, W2w = sol["X"], sol["W2"]
            u_prev, w2_prev = u0.copy(), w2_0.copy()
        else:
            u0 = u_prev

        u_dev = u0 - ur
        u_dev_feas, omega = apply_rpm_saturation(
            u_dev, u_eq=ur, params=params,
            f_u2w2=omegas2_from_thrust_torques,
            f_w22u=thrust_torques_from_omegas2,
            clip_fun=clip_omegas
        )
        u_applied = ur + u_dev_feas
        if np.any(np.isclose(omega, params.min_rpm)) or np.any(np.isclose(omega, params.max_rpm)):
            sat_steps += 1

        x = rk4_step(lambda xx,uu: f_dynamics(xx,uu,params,u_mode="Ttau"), x, u_applied, dt)
        X[k]=x; U[k]=u_applied; O[k]=omega

        if k % max(1,int(round(0.5/dt)))==0 or k==n_steps-1:
            ep = float(np.linalg.norm(x[:3]-xr[:3])); ez = float(x[2]-xr[2])
            eyaw = float(np.degrees(wrap_pi(x[8]-xr[8])))
            pct = 100.0*k/(n_steps-1)
            print(f"\r[N={N:3d}] {pct:5.1f}% t={T[k]:.2f}/{T[-1]:.2f}s | ||e_p||={ep:.3f}, ez={ez:+.3f}, eyaw={eyaw:+.2f}°, sat%={100*sat_steps/max(1,k+1):.1f}", end="", flush=True)
    print("")

    XR_full = np.tile(xr.reshape(1,-1),(X.shape[0],1))
    E = X - XR_full; E[:,8] = np.array([wrap_pi(e) for e in E[:,8]])
    J = ct_cost(T, X, U - ur, Q, R, XR_full)
    idxs, ts_settle, os_pct = step_metrics_multi(T, X, xr, indices=(0,1,2,6,7,8), tol=0.02)
    peak_abs = np.array([np.max(np.abs(E[:,i])) for i in (0,1,2,8)])
    sat_frac = float(sat_steps)/float(max(1, X.shape[0]))
    return dict(status="solved", N=int(N), dt_ctrl=float(dt_ctrl), dt_plant=float(dt),
                T=T, X=X, U=U, O=O, E=E, x_eq=xr*0+xr, u_eq=ur, xr=xr,
                J=float(J), idxs=np.array((0,1,2,8)), ts_settle=ts_settle.astype(float),
                os_pct=os_pct.astype(float), peak_abs=peak_abs.astype(float),
                sat_steps=int(sat_steps), sat_frac=float(sat_frac))

def main():
    apply_style(preset="pub")
    params = QuadParams()
    A,B,x_eq,ur = linearize_hover(params, yaw=0.0)
    xr = _xref(x_eq); x0 = _x0(x_eq)
    dt = float(DT)
    dt_ctrl = max(1, int(round(DT_CTRL/dt))) * dt

    results_dir = os.path.join(SIM_DIR, RESULTS_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")
    summary=[]
    for N in N_LIST:
        res = run_one(N, params, dt, dt_ctrl, x0, xr, ur)
        fbase = f"nmpcN{int(N):03d}_dt{int(round(1000*dt_ctrl)):03d}"
        npz_path = os.path.join(results_dir, fbase + ".npz")
        save_npz_package(npz_path, res, {"case": f"N={N}", "status": res["status"]})
        print(f"saved -> {npz_path}  [{res['status']}]")
        summary.append((N, res["status"], res.get("J", np.nan), res.get("sat_frac", np.nan)))
    print("\n=== Summary ===")
    for N, st, J, sf in summary:
        print(f"N={N:3d}  status={st:<16s}  J={J:9.3f}  sat%={100*sf:6.2f}")

if __name__ == "__main__":
    main()
