#!/usr/bin/env python3
from __future__ import annotations
"""
Quadrotor dynamics and allocation (NED frame, ZYX Euler angles).

State:  x = [p(3), v(3), η(3: φ,θ,ψ), ω(3: p,q,r)] ∈ R^12
Input:  u = [T, τx, τy, τz]  (default mode "Ttau")

Conventions
-----------
- NED frame: +z is down; gravity points +z.
- Euler sequence ZYX (yaw–pitch–roll).
- Thrust acts along body -z; v̇ = g e3 - (T/m) (R e3).
- "X" motor configuration.

Project-generic helpers live in utils.py; constants in macros.py.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np

# Utils
from utils import finite_difference_jacobian as _fdjac, R_zyx as _R_zyx, W_zyx as _W_zyx

# Centralized constants
from macros import (
    M as MASS, G, JX, JY, JZ, LX, LY, KF, CTAU, MAX_RPM, MIN_RPM, LIN_EPS_X, LIN_EPS_U
)

# Parameters

@dataclass
class QuadParams:
    """Physical, geometric, and actuator parameters for the quadrotor."""
    m: float = MASS
    g: float = G
    Jx: float = JX
    Jy: float = JY
    Jz: float = JZ
    Lx: float = LX
    Ly: float = LY
    kf: float = KF
    ctau: float = CTAU
    max_rpm: float = MAX_RPM
    min_rpm: float = MIN_RPM
    lin_eps_x: float = LIN_EPS_X
    lin_eps_u: float = LIN_EPS_U

    @property
    def J(self) -> np.ndarray:
        return np.diag([self.Jx, self.Jy, self.Jz])

    @property
    def e3(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

# Allocation

def allocation_matrices(p: QuadParams) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Return (M, M_inv) such that:
      [T, τx, τy, τz]^T = M [Ω1^2, Ω2^2, Ω3^2, Ω4^2]^T
      [Ω1^2, Ω2^2, Ω3^2, Ω4^2]^T = M_inv [T, τx, τy, τz]^T

    'X' configuration (roll uses Ly, pitch uses Lx):
      τx row: [-, -, +, +] with lever Ly/2
      τy row: [+, -, -, +] with lever Lx/2
      τz row: [+, -, +, -]
    """
    kf, ctau, Lx, Ly = p.kf, p.ctau, p.Lx, p.Ly
    M = np.array([
        [ kf,           kf,           kf,           kf          ],
        [-kf*Ly/2.0,   -kf*Ly/2.0,    kf*Ly/2.0,    kf*Ly/2.0   ],
        [ kf*Lx/2.0,   -kf*Lx/2.0,   -kf*Lx/2.0,    kf*Lx/2.0   ],
        [ kf*ctau,     -kf*ctau,      kf*ctau,     -kf*ctau    ],
    ], dtype=float)
    M_inv = np.linalg.inv(M)
    return M, M_inv

def thrust_torques_from_omegas2(omega2: np.ndarray, p: QuadParams) -> np.ndarray:
    """Map [Ω1^2..Ω4^2] (RPM^2) → [T, τx, τy, τz]."""
    M, _ = allocation_matrices(p)
    return M @ np.asarray(omega2, float)

def omegas2_from_thrust_torques(Ttau: np.ndarray, p: QuadParams) -> np.ndarray:
    """Map [T, τx, τy, τz] → [Ω1^2..Ω4^2] (RPM^2); negatives clipped to zero."""
    _, M_inv = allocation_matrices(p)
    omega2 = M_inv @ np.asarray(Ttau, float)
    return np.maximum(omega2, 0.0)

def clip_omegas(omega: np.ndarray, p: QuadParams) -> np.ndarray:
    """Clip rotor speeds [Ω1..Ω4] (RPM) to [min_rpm, max_rpm]."""
    return np.clip(np.asarray(omega, float), p.min_rpm, p.max_rpm)

# Nonlinear dynamics

def hover_equilibrium(
    p: QuadParams, position: Optional[np.ndarray] = None, yaw: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hover equilibrium (x0, u0) with φ=θ=0, ψ=yaw, v=0, ω=0, T0=mg, τ=0.
    State x = [p(3), v(3), η(3), ω(3)], Input u = [T, τx, τy, τz].
    """
    if position is None:
        position = np.zeros(3)
    x0 = np.zeros(12)
    x0[0:3] = position
    x0[8] = float(yaw)
    u0 = np.array([p.m * p.g, 0.0, 0.0, 0.0], dtype=float)
    return x0, u0

def f_dynamics(
    x: np.ndarray, u: np.ndarray, p: QuadParams, u_mode: Literal["Ttau", "omega2"] = "Ttau"
) -> np.ndarray:
    """
    Continuous-time dynamics (NED, ZYX). If u_mode=="Ttau": u=[T,τ], else u=[Ω^2].
    """
    px, py, pz, vx, vy, vz, phi, theta, psi, p_body, q_body, r_body = np.asarray(x, float)
    v = np.array([vx, vy, vz]); omega = np.array([p_body, q_body, r_body])
    # Inputs → (T, τ)
    if u_mode == "Ttau":
        Ttau = np.asarray(u, float)
    elif u_mode == "omega2":
        Ttau = thrust_torques_from_omegas2(np.asarray(u, float), p)
    else:
        raise ValueError("u_mode must be 'Ttau' or 'omega2'")
    T, taux, tauy, tauz = Ttau; tau = np.array([taux, tauy, tauz])

    # Kinematics
    R = _R_zyx(phi, theta, psi)
    W = _W_zyx(phi, theta)

    # Translational: v̇ = g e3 - (T/m) (R e3)
    pdot = v
    vdot = p.g * p.e3 - (T / p.m) * (R @ p.e3)

    # Rotational: η̇ = W ω ;  ω̇ = J^{-1} (τ - ω×Jω)
    J = p.J
    etadot = W @ omega
    omegadot = np.linalg.solve(J, tau - np.cross(omega, J @ omega))

    dx = np.zeros(12)
    dx[0:3] = pdot; dx[3:6] = vdot; dx[6:9] = etadot; dx[9:12] = omegadot
    return dx

# Linearization around hover

def linearize_hover(p: QuadParams, yaw: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerical linearization at hover (φ=θ=0, ψ=yaw, v=0, ω=0, T=mg, τ=0).
    Returns (A, B, x0, u0_abs) where u = [T, τx, τy, τz].
    """
    x0, u0_abs = hover_equilibrium(p, yaw=yaw)
    fx = lambda xx: f_dynamics(xx, u0_abs, p, u_mode="Ttau")
    fu = lambda uu: f_dynamics(x0, uu, p, u_mode="Ttau")
    A = _fdjac(fx, x0, eps=p.lin_eps_x)
    B = _fdjac(fu, u0_abs, eps=p.lin_eps_u)
    return A, B, x0, u0_abs

# Quick CLI check
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=False)
    p = QuadParams()
    M, M_inv = allocation_matrices(p)
    print("M:\n", M, "\nM_inv:\n", M_inv, "\n")
    A, B, x0, u0 = linearize_hover(p, yaw=0.0)
    print("x0:", x0, "\nu0:", u0)
    print("A:\n", A, "\nB:\n", B)
