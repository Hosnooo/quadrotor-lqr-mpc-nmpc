from __future__ import annotations
import numpy as np

__all__ = [
    "rot_kinematics_blocks",         # R[:,3] and its derivatives + E and its derivatives
    "analytic_jacobians",  # A(x,u), G(x,u) for the 12-state, 4-input model
]

# Trig helpers (guard near pitch singularity)                                   #

def _euler_trig(phi: float, theta: float, psi: float):
    """
    Return (sphi, cphi, sth, cth, sps, cps, tth, sec_th) with a small guard
    to avoid |cos(theta)| ~ 0 singularity in ZYX (roll-pitch-yaw).
    """
    sphi, cphi = np.sin(phi), np.cos(phi)
    sth,  cth  = np.sin(theta), np.cos(theta)
    sps,  cps  = np.sin(psi),   np.cos(psi)

    # guard away from |cos(theta)| ~ 0 to avoid singularity in E(φ,θ)
    eps = 1e-8
    if abs(cth) < eps:
        cth = np.copysign(eps, cth)

    tth    = sth / cth
    sec_th = 1.0 / cth
    return sphi, cphi, sth, cth, sps, cps, tth, sec_th

# Rotation/kinematics                                   #

def rot_kinematics_blocks(phi: float, theta: float, psi: float):
    """
    Compute and return the blocks that depend on (φ,θ,ψ) once:

        R3        := third column of Rz(ψ) Ry(θ) Rx(φ)  (body z in world)
        dR3_dφ, dR3_dθ, dR3_dψ
        E(φ,θ)    := Euler kinematics matrix (ZYX)
        dE_dφ, dE_dθ

    Returns
    -------
    R3 : (3,)
    dR3_dphi : (3,)
    dR3_dtheta : (3,)
    dR3_dpsi : (3,)
    E : (3,3)
    dE_dphi : (3,3)
    dE_dtheta : (3,3)
    """
    sphi, cphi, sth, cth, sps, cps, tth, sec_th = _euler_trig(phi, theta, psi)

    # R third column (world z-axis expressed in body? -> here: body z in world)
    R3 = np.array([
        cps * sth * cphi + sps * sphi,
        sps * sth * cphi - cps * sphi,
        cth * cphi
    ], dtype=float)

    # d/dφ R3
    dR3_dphi = np.array([
        -cps * sth * sphi + sps * cphi,
        -sps * sth * sphi - cps * cphi,
        -cth * sphi
    ], dtype=float)

    # d/dθ R3
    dR3_dtheta = np.array([
        cps * cth * cphi,
        sps * cth * cphi,
        -sth * cphi
    ], dtype=float)

    # d/dψ R3
    dR3_dpsi = np.array([
        -sps * sth * cphi + cps * sphi,
         cps * sth * cphi + sps * sphi,
         0.0
    ], dtype=float)

    # E(φ,θ)
    E = np.array([
        [1.0,          sphi * tth,        cphi * tth],
        [0.0,          cphi,              -sphi     ],
        [0.0,          sphi * sec_th,     cphi * sec_th]
    ], dtype=float)

    # dE/dφ
    dE_dphi = np.array([
        [0.0,  cphi * tth,       -sphi * tth],
        [0.0, -sphi,             -cphi      ],
        [0.0,  cphi * sec_th,    -sphi * sec_th]
    ], dtype=float)

    # dE/dθ (using d/dθ tanθ = sec^2θ and d/dθ secθ = secθ tanθ)
    sec2 = sec_th * sec_th
    dE_dtheta = np.array([
        [0.0,  sphi * sec2,           cphi * sec2        ],
        [0.0,  0.0,                   0.0                ],
        [0.0,  sphi * sec_th * tth,   cphi * sec_th * tth]
    ], dtype=float)

    return R3, dR3_dphi, dR3_dtheta, dR3_dpsi, E, dE_dphi, dE_dtheta

# Full analytic Jacobians A(x,u), G(x,u)                                        #

def analytic_jacobians(x: np.ndarray, u: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytic Jacobians for the 12-state quadrotor with ZYX Euler and inputs [T, τx, τy, τz].

    State/order (Section 1/2):
        x = [p_x, p_y, p_z, v_x, v_y, v_z, φ, θ, ψ, p, q, r]
        u = [T, τx, τy, τz]

    Dynamics:
        ṗ = v
        v̇ = [0,0,-g]^T - (T/m) * R3(φ,θ,ψ)
        [φ̇, θ̇, ψ̇]^T = E(φ,θ) * [p, q, r]^T
        ω̇ = J^{-1} (τ - ω × Jω), ω = [p, q, r]^T, J = diag(Jx, Jy, Jz)

    Returns
    -------
    A : (12, 12) = ∂f/∂x
    G : (12,  4) = ∂f/∂u
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float).reshape(4,)

    m  = float(params.m)
    Jx = float(params.Jx)
    Jy = float(params.Jy)
    Jz = float(params.Jz)

    # Unpack state, input
    px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r = x
    T, tx, ty, tz = u

    # Rotation/kinematics blocks (computed ONCE)
    R3, dR3_dphi, dR3_dtheta, dR3_dpsi, E, dE_dphi, dE_dtheta = rot_kinematics_blocks(phi, theta, psi)
    omega = np.array([p, q, r], dtype=float)

    # Initialize A, G
    A = np.zeros((12, 12), dtype=float)
    G = np.zeros((12,  4), dtype=float)

    # ṗ = v
    A[0:3, 3:6] = np.eye(3)

    # v̇ = [0,0,-g] - (T/m) * R3
    A[3:6, 6] = -(T / m) * dR3_dphi
    A[3:6, 7] = -(T / m) * dR3_dtheta
    A[3:6, 8] = -(T / m) * dR3_dpsi
    # ∂v̇/∂T
    G[3:6, 0] = -(1.0 / m) * R3

    # η̇ = E(φ,θ) * ω
    A[6:9, 6]    = dE_dphi @ omega
    A[6:9, 7]    = dE_dtheta @ omega
    A[6:9, 9:12] = E

    # ω̇ = J^{-1}(τ - ω×Jω)
    # With diagonal J, ∂(ω×Jω)/∂ω yields the antisymmetric entries below.
    A[9, 10]  = -((Jz - Jy) / Jx) * r   # ∂dot p / ∂q
    A[9, 11]  = -((Jz - Jy) / Jx) * q   # ∂dot p / ∂r

    A[10, 9]  = -((Jx - Jz) / Jy) * r   # ∂dot q / ∂p
    A[10, 11] = -((Jx - Jz) / Jy) * p   # ∂dot q / ∂r

    A[11, 9]  = -((Jy - Jx) / Jz) * q   # ∂dot r / ∂p
    A[11, 10] = -((Jy - Jx) / Jz) * p   # ∂dot r / ∂q

    # ∂ω̇/∂τ
    G[9, 1]  = 1.0 / Jx
    G[10, 2] = 1.0 / Jy
    G[11, 3] = 1.0 / Jz

    return A, G

# Finite-difference check

def _fd_jacobians_once(x: np.ndarray, u: np.ndarray, params,
                       f_rhs, eps_x: float = 1e-6, eps_u: float = 1e-6):
    """
    Central-difference A,G at (x,u) for a given RHS f_rhs(x,u)->xdot.
    Returns:
      A_fd (12x12) ≈ ∂f/∂x via (f(x+dx) - f(x-dx)) / (2*eps_x)
      G_fd (12x4)  ≈ ∂f/∂u via (f(u+du) - f(u-du)) / (2*eps_u)

    Notes
    -----
    - Central differences are O(eps^2) accurate (vs O(eps) for forward).
    - Keep eps_x, eps_u small but not too small (avoid cancellation). 1e-6–1e-5
      is typically robust for this model.
    """
    x = np.asarray(x, float).copy()
    u = np.asarray(u, float).copy().reshape(4,)

    nx, nu = x.size, u.size
    A_fd = np.zeros((nx, nx))
    G_fd = np.zeros((nx, nu))

    # Columns of A by perturbing x (central difference)
    for i in range(nx):
        dx = np.zeros_like(x)
        dx[i] = eps_x
        f_plus  = np.asarray(f_rhs(x + dx, u), float)
        f_minus = np.asarray(f_rhs(x - dx, u), float)
        A_fd[:, i] = (f_plus - f_minus) / (2.0 * eps_x)

    # Columns of G by perturbing u (central difference)
    for j in range(nu):
        du = np.zeros_like(u)
        du[j] = eps_u
        f_plus  = np.asarray(f_rhs(x, u + du), float)
        f_minus = np.asarray(f_rhs(x, u - du), float)
        G_fd[:, j] = (f_plus - f_minus) / (2.0 * eps_u)

    return A_fd, G_fd

def check_analytic_jacobians(params, seed: int = 0):
    """
    One-shot check at a hover-near state/input:
      - builds x,u near hover,
      - compares analytic A,G to finite-difference A_fd,G_fd,
      - prints max abs/rel errors.
    """
    import numpy as _np
    from dynamics import linearize_hover, f_dynamics  # local import to avoid cycles

    _rng = _np.random.default_rng(seed)

    # hover linearization gives (x0,u0); f_dynamics expects absolute T, τx, τy, τz
    A_lin, B_lin, x0, u0 = linearize_hover(params, yaw=0.0)

    # small random perturbation around hover to avoid trivial zero derivatives
    x = x0.copy()
    x[0:3] += 1e-3 * _rng.standard_normal(3)      # position
    x[3:6] += 1e-3 * _rng.standard_normal(3)      # velocity
    x[6:9] += 1e-3 * _rng.standard_normal(3)      # small angles
    x[9:12] += 1e-3 * _rng.standard_normal(3)     # body rates

    u = u0.copy()
    u += _np.array([0.01, 0.001, -0.001, 0.001])   # tiny nonzero torques

    def _rhs(xx, uu):
        return f_dynamics(xx, uu, params, u_mode="Ttau")

    # analytic
    A_an, G_an = analytic_jacobians(x, u, params)
    # finite difference
    A_fd, G_fd = _fd_jacobians_once(x, u, params, _rhs)

    abs_err_A = _np.max(_np.abs(A_an - A_fd))
    abs_err_G = _np.max(_np.abs(G_an - G_fd))

    # relative error w.r.t. 1 + |fd|
    rel_err_A = _np.max(_np.abs(A_an - A_fd) / (1.0 + _np.abs(A_fd)))
    rel_err_G = _np.max(_np.abs(G_an - G_fd) / (1.0 + _np.abs(G_fd)))

    print("[analytic Jacobians check]")
    print(f"  max |A_an - A_fd| = {abs_err_A:.3e},   max rel = {rel_err_A:.3e}")
    print(f"  max |G_an - G_fd| = {abs_err_G:.3e},   max rel = {rel_err_G:.3e}")

if __name__ == "__main__":
    from dynamics import QuadParams
    check_analytic_jacobians(QuadParams())
