# casadi_ms.py
from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
import casadi as ca

from dynamics import thrust_torques_from_omegas2, QuadParams
from macros import LIN_EPS_X  # used only if you rely on it elsewhere

# ------------------------------------------------------------
# CasADi symbols: rotation & maps (ZYX)
# ------------------------------------------------------------
def R_zyx_sx(phi, theta, psi):
    cφ, sφ = ca.cos(phi), ca.sin(phi)
    cθ, sθ = ca.cos(theta), ca.sin(theta)
    cψ, sψ = ca.cos(psi), ca.sin(psi)
    Rz = ca.vertcat(
        ca.hcat([ cψ, -sψ, 0]),
        ca.hcat([ sψ,  cψ, 0]),
        ca.hcat([  0,   0, 1]),
    )
    Ry = ca.vertcat(
        ca.hcat([ cθ, 0, sθ]),
        ca.hcat([  0, 1,  0]),
        ca.hcat([-sθ, 0, cθ]),
    )
    Rx = ca.vertcat(
        ca.hcat([1,  0,   0]),
        ca.hcat([0, cφ, -sφ]),
        ca.hcat([0, sφ,  cφ]),
    )
    return ca.mtimes(Rz, ca.mtimes(Ry, Rx))

def W_zyx_sx(phi, theta):
    sφ, cφ = ca.sin(phi), ca.cos(phi)
    sθ, cθ = ca.sin(theta), ca.cos(theta)
    return ca.vertcat(
        ca.hcat([1,    sφ*ca.tan(theta),  cφ*ca.tan(theta)]),
        ca.hcat([0,               cφ,             -sφ     ]),
        ca.hcat([0,   sφ/cθ,                 cφ/cθ       ]),
    )

# ------------------------------------------------------------
# Symbolic dynamics + RK4 one-step map Φ(x, W2; h)
# ------------------------------------------------------------
def make_symbolic_dynamics(params: QuadParams):
    m, g = params.m, params.g
    Jx, Jy, Jz = params.Jx, params.Jy, params.Jz
    Lx, Ly, kf, ctau = params.Lx, params.Ly, params.kf, params.ctau

    # constant allocation M: maps W2 -> [T, τx, τy, τz]
    M = ca.DM([
        [ kf,       kf,       kf,       kf     ],
        [-kf*Ly/2, -kf*Ly/2,  kf*Ly/2,  kf*Ly/2],
        [ kf*Lx/2, -kf*Lx/2, -kf*Lx/2,  kf*Lx/2],
        [ kf*ctau, -kf*ctau,  kf*ctau, -kf*ctau],
    ])

    x  = ca.SX.sym('x', 12)     # [p(3), v(3), (φ,θ,ψ), (p,q,r)]
    W2 = ca.SX.sym('w2', 4)     # Ω^2 control
    h  = ca.SX.sym('h')

    # unpack
    p      = x[0:3]
    v      = x[3:6]
    phi    = x[6]
    theta  = x[7]
    psi    = x[8]
    omega  = x[9:12]

    Ttau = ca.mtimes(M, W2)
    T    = Ttau[0]
    tau  = Ttau[1:4]

    R = R_zyx_sx(phi, theta, psi)
    W = W_zyx_sx(phi, theta)

    e3 = ca.DM([0, 0, 1])
    pdot   = v
    vdot   = g*e3 - (T/m) * ca.mtimes(R, e3)          # NED (+z down)
    etadot = ca.mtimes(W, omega)

    Jmat   = ca.diag(ca.DM([Jx, Jy, Jz]))
    Jinv   = ca.diag(ca.DM([1.0/Jx, 1.0/Jy, 1.0/Jz]))
    cross  = ca.cross(omega, ca.mtimes(Jmat, omega))
    omegadot = ca.mtimes(Jinv, (tau - cross))

    xdot = ca.vertcat(pdot, vdot, etadot, omegadot)

    # continuous-time f(x,W2)
    f_dyn = ca.Function('f_dyn', [x, W2], [xdot], ['x','w2'], ['xdot'])

    # RK4 step
    k1 = f_dyn(x,              W2)
    k2 = f_dyn(x + 0.5*h*k1,   W2)
    k3 = f_dyn(x + 0.5*h*k2,   W2)
    k4 = f_dyn(x +     h*k3,   W2)
    xnext = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # one-step map and its "u" (T,τ)
    Phi = ca.Function('Phi', [x, W2, h], [xnext, Ttau], ['x','w2','h'], ['xnext','u'])

    # exact jacobians of xnext wrt x and w2 (used for PMP checks)
    Fx = ca.jacobian(xnext, x)
    Fw2 = ca.jacobian(xnext, W2)
    PhiJ = ca.Function('PhiJ', [x, W2, h], [Fx, Fw2], ['x','w2','h'], ['Fx','Fw2'])

    return Phi, PhiJ, M

# ------------------------------------------------------------
# OCP (direct multiple shooting) + costates / PMP residuals
# ------------------------------------------------------------
def ocp_casadi_ms_ipopt(
    params: QuadParams,
    x0: np.ndarray,
    t0: float,
    tf: float,
    dt: float,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
    x_ref: Callable[[float], np.ndarray],
    u_ref: Callable[[float], np.ndarray],
    W2_init: np.ndarray | None = None,
    ipopt_max_iter: int = 400,
    ipopt_verb: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Direct Multiple Shooting with hard bounds on W2 (Ω²).
    Decision vars: X_k (k=0..N-1), W2_k (k=0..N-2), constraints X_{k+1}=Φ(X_k,W2_k).
    Cost: 0.5∑(dxᵀQdx + duᵀRdu)·dt + 0.5 (xN-xr)ᵀQf(xN-xr).
    Returns (T, X, U, O, info) — with PMP artifacts inside info.
    """
    # grid
    N = int(np.round((tf - t0)/dt)) + 1
    T = np.linspace(t0, tf, N)
    nx, nu = x0.size, 4

    # bounds from params
    wmin2 = float(params.min_rpm)**2
    wmax2 = float(params.max_rpm)**2
    if not np.isfinite(wmin2) or not np.isfinite(wmax2) or wmax2 <= wmin2:
        raise ValueError("Bad RPM bounds from params.")

    # maps, jacobians
    Phi, PhiJ, M = make_symbolic_dynamics(params)

    # decision arrays
    X  = ca.SX.sym('X',  nx, N)        # states at nodes 0..N-1 (we'll add X_N as X[:, -1])
    W2 = ca.SX.sym('W2', nu, N-1)      # control per interval

    Qm  = ca.DM(Q);  Rm = ca.DM(R);  Qfm = ca.DM(Qf)
    xr  = ca.DM(x_ref(0.0))            # constant refs
    ur  = ca.DM(u_ref(0.0))

    J = 0
    g = []   # constraints = [X0 - x0, X1 - Φ(X0,W2_0), ..., XN - Φ(XN-1,W2_{N-2})]
    lbg = []
    ubg = []

    # initial condition
    g += [X[:,0] - ca.DM(x0)]
    lbg += [0]*nx; ubg += [0]*nx

    # shooting constraints + stage cost
    for k in range(N-1):
        xk  = X[:,k]
        w2k = W2[:,k]
        xk1 = X[:,k+1]

        xnext, uk = Phi(xk, w2k, dt)
        g += [xk1 - xnext]
        lbg += [0]*nx; ubg += [0]*nx

        dx = xk - xr
        du = uk - ur
        J += 0.5 * (ca.mtimes([dx.T, Qm, dx]) + ca.mtimes([du.T, Rm, du])) * dt

    # terminal
    dxT = X[:, -1] - xr
    J  += 0.5 * ca.mtimes([dxT.T, Qfm, dxT])

    # decision vector
    w = ca.vertcat(X.reshape((-1,1)), W2.reshape((-1,1)))

    # bounds on decision vars
    lbw = []; ubw = []
    # X free (X0 fixed by equality constraint)
    lbw += [-ca.inf]*(nx*N); ubw += [ ca.inf]*(nx*N)
    # W2 box bounds
    for _ in range(N-1):
        lbw += [wmin2]*nu; ubw += [wmax2]*nu

    # initial guess
    if W2_init is None:
        W2_init = np.full((N-1, nu), 0.5*(wmin2 + wmax2), float)
    w0 = np.zeros((nx*N + (N-1)*nu,))
    # X guess: hover
    for k in range(N):
        w0[k*nx:(k+1)*nx] = x0
    # W2 guess
    off = nx*N
    w0[off:off+(N-1)*nu] = W2_init.reshape(-1)

    # solver
    nlp = {'x': w, 'f': J, 'g': ca.vertcat(*g)}
    opts = {
        'ipopt.print_level': ipopt_verb,
        'ipopt.max_iter': ipopt_max_iter,
        'ipopt.sb': 'yes',
        'print_time': True,
        'ipopt.tol': 1e-8,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.linear_solver': 'mumps',
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    w_opt  = np.array(sol['x']).squeeze()
    lam_g  = np.array(sol['lam_g']).squeeze()
    info   = dict(solver.stats())

    # unpack X, W2
    X_opt  = w_opt[:nx*N].reshape(nx, N, order='F')
    W2_opt = w_opt[nx*N:].reshape(nu, N-1, order='F')

    # outputs for plotting/alignment
    U_nm1   = (M @ ca.DM(W2_opt)).full().T           # (N-1,4)
    U       = np.vstack([U_nm1, U_nm1[-1]])          # (N,4)
    W2_full = np.vstack([W2_opt.T, W2_opt.T[-1]])    # (N,4)
    O       = np.sqrt(np.maximum(W2_full, 0.0))      # RPM

    # --------------------------------------------------------
    # PMP artifacts (discrete): costates & residual checks
    # --------------------------------------------------------
    # multipliers for equality constraints g = [IC (nx), then N-1 defects]
    if N >= 2:
        lam_dyn = lam_g[nx:].reshape(N-1, nx)        # multiplier block per defect -> λ_{k+1}
    else:
        lam_dyn = np.zeros((0, nx))

    # backward recursion using exact Jacobians of one-step map
    lam_hat = np.zeros_like(lam_dyn)                 # λ̂_{k+1}
    ru_list = []                                     # stationarity residual wrt W2
    rp_list = []                                     # consistency with IPOPT multipliers

    lamN = (Qfm @ (X_opt[:, -1] - xr)).full().squeeze()
    lam_next = lamN.copy()

    for k in reversed(range(N-1)):
        xk  = X_opt[:, k]
        w2k = W2_opt[:, k]
        Fx, Fw2 = PhiJ(xk, w2k, dt)                  # exact discrete jacobians
        Fx   = np.array(Fx)
        Fw2  = np.array(Fw2)

        uk = (M @ w2k.reshape(-1,1)).full().squeeze()
        du = uk - np.array(ur).squeeze()

        # Stationarity wrt W2: Mᵀ R (u_k - u_ref) * dt + F_w2ᵀ λ_{k+1} ≈ 0
        ru_k = (M.T @ (R @ du)) * dt + Fw2.T @ lam_next
        ru_list.append(ru_k)

        # λ̂_{k+1}
        lam_hat[k, :] = lam_next

        # Consistency with IPOPT multipliers (lam_dyn[k] ≈ λ_{k+1})
        rp_k = lam_dyn[k, :] - lam_next
        rp_list.append(rp_k)

        # backward update: λ_k = Fxᵀ λ_{k+1} + Q (x_k - x_ref) * dt
        lam_next = Fx.T @ lam_next + (Q @ (xk - np.array(xr).squeeze())) * dt

    ru = np.array(ru_list[::-1]) if N > 1 else np.zeros((0, nu))
    rp = np.array(rp_list[::-1]) if N > 1 else np.zeros((0, nx))

    metrics = {
        "ru_max_inf": float(np.max(np.linalg.norm(ru, ord=np.inf, axis=1))) if ru.size else 0.0,
        "rp_max_inf": float(np.max(np.linalg.norm(rp, ord=np.inf, axis=1))) if rp.size else 0.0,
        "term_err_inf": float(np.linalg.norm(lam_dyn[-1, :] - lamN, ord=np.inf)) if (N > 1) else 0.0,
    }

    # stash PMP artifacts into info
    info.update({
        "lam_dyn": lam_dyn,       # (N-1, nx) multipliers on dynamics defects = λ_{k+1}
        "lam_hat": lam_hat,       # (N-1, nx) from backward recursion
        "ru": ru,                 # (N-1, 4)  stationarity residuals wrt W2
        "rp": rp,                 # (N-1, nx) consistency residuals
        "lamN": lamN,             # (nx,)
        "pmp_metrics": metrics,
    })

    return T, X_opt.T.copy(), U.copy(), O.copy(), info



def main():
    pass


if __name__ == "__main__":
    main()
