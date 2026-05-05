#!/usr/bin/env python3
from __future__ import annotations
"""
macros.py
Central registry for project-wide constants and tiny helpers.

Conventions
-----------
- NED frame (+z down), Euler Z-Y-X.
- State: x = [p(3), v(3), φ, θ, ψ, p, q, r] ∈ R^12
- Input: u = [T, τx, τy, τz]
"""

import os
import numpy as np

# Paths / naming

ARTIFACTS_DIR = "Artifacts"     # where controller artifacts (LQR/TV-LQR) are stored
SIM_DIR       = "Simulations"   # where simulation NPZs/figures are written

LQR_PREFIX = "lqr_hover_"       # controller package files start with this
SIM_PREFIX = {
    "INF":        "lqr_",            # static LQR∞ rollout
    "TV_LOCAL":   "lqr_TV_local_reset_", # tv-lqr with per-switch reset
    "TV_GLOBAL":  "lqr_TV_global_time_", # tv-lqr single timebase
}

# Physics / geometry / actuation

# Mass / gravity
M = 1.505        # [kg]
G = 9.80665      # [m/s^2]

# Principal inertias (diagonal J)
JX = 0.03        # [kg·m^2]
JY = 0.03
JZ = 0.06

# Arm lengths (roll lever uses Ly, pitch lever uses Lx)
LX = 0.2032      # [m]  (≈ 8 in)
LY = 0.2540      # [m]  (≈ 10 in)

# Motor/prop coefficients
KF   = 2.142e-8  # [N / RPM^2]
CTAU = 1.451e-2  # yaw moment coefficient ratio

# Actuator bounds
MAX_RPM = 20000.0
MIN_RPM = 0.0

# Numerics (integration / diffs)

DT        = 0.002   # [s] global nominal simulation step
LIN_EPS_X = 1e-7    # finite-difference step for state
LIN_EPS_U = 1e-6    # finite-difference step for input

# Mission / setpoint defaults

# Default multi-waypoint mission (absolute NED; "up 3 m" => pz = -3)
MISSION = [
    (0.0, 0.0, -2.0, 0.0),
    (1.0, 0.0, -2.0, 0.0),
    (1.0, 1.0, -3.0, 0.0),
    (0.0, 1.0, -3.0, 0.0),
    (0.0, 0.0, -2.0, 0.0),
]
HOLD_SEC = 5.0     # [s] duration per waypoint
RAMP_SEC = 0.75    # [s] optional min-jerk smoothing (0.0 -> hard step)

# Single-setpoint run (PMP helper)
T_SINGLE_SETPOINT = 8.0
SETPOINT = dict(px=1.0,  py=-1.0, pz=-3.0, yaw=0.05, pitch=0.0, roll=0.0)
OFFSETS  = dict(px0=0.25, py0=-0.25, pz0=0.0)

# PMP solver defaults

PMP_MAX_ITERS = 30
PMP_ALPHA     = 0.5
PMP_TOL       = 1e-4

# Canonical LQR weights (fallbacks)

Q_DIAG = np.array([
    10.0, 10.0, 12.0,   # position
     2.0,  2.0,  2.0,   # velocity
     8.0,  8.0,  2.0,   # angles (φ,θ,ψ)
     0.5,  0.5,  0.5    # body rates (p,q,r)
], dtype=float)

R_DIAG = np.array([1.0, 0.05, 0.05, 0.02], dtype=float)  # [T, τx, τy, τz]

def Q_matrix(diag: np.ndarray | None = None) -> np.ndarray:
    d = Q_DIAG if diag is None else np.asarray(diag, float)
    if d.size != 12:
        raise ValueError("Q diag must have 12 entries")
    return np.diag(d)

def R_matrix(diag: np.ndarray | None = None) -> np.ndarray:
    d = R_DIAG if diag is None else np.asarray(diag, float)
    if d.size != 4:
        raise ValueError("R diag must have 4 entries")
    return np.diag(d)

# Tiny shared helper

def newest_path(folder: str, prefix: str, suffix: str = ".npz") -> str | None:
    """Newest file in folder matching prefix/suffix, else None."""
    if not os.path.isdir(folder):
        return None
    cands = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.startswith(prefix) and f.endswith(suffix)]
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]



def main():
    pass


if __name__ == "__main__":
    main()
