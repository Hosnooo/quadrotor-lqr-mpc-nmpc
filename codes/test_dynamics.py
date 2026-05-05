#!/usr/bin/env python3
from __future__ import annotations
"""Unit checks for allocation, equilibrium, linearization, and round-trips."""

import numpy as np
from dynamics import (
    QuadParams, allocation_matrices, thrust_torques_from_omegas2,
    omegas2_from_thrust_torques, hover_equilibrium, f_dynamics, linearize_hover
)

np.set_printoptions(suppress=False, linewidth=160)

def almost_equal(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol, rtol=0.0)

def main():
    p = QuadParams()  # Lx=0.2032, Ly=0.254, kf=2.142e-8, ctau=0.01451
    M, M_inv = allocation_matrices(p)

    # 1) Allocation / inverse consistency
    I4 = M @ M_inv
    print("M:\n", M)
    print("\nM_inv:\n", M_inv)
    print("\nM @ M_inv:\n", I4)
    assert almost_equal(I4, np.eye(4), tol=1e-8), "M * M_inv should be identity (within 1e-8)."
    print("Allocation inverse consistency passed.\n")

    # 2) Numeric magnitudes (roll/pitch/yaw rows)
    kf, Lx, Ly, ctau = p.kf, p.Lx, p.Ly, p.ctau
    expected_roll  = kf * Ly / 2.0       # ≈ 2.720e-9
    expected_pitch = kf * Lx / 2.0       # ≈ 2.176e-9
    expected_yaw   = kf * ctau           # ≈ 3.108e-10
    assert np.isclose(abs(M[1, 2]), expected_roll,  rtol=0, atol=1e-12)
    assert np.isclose(abs(M[2, 0]), expected_pitch, rtol=0, atol=1e-12)
    assert np.isclose(abs(M[3, 0]), expected_yaw,   rtol=0, atol=1e-14)
    print(f"Row magnitudes OK: roll={expected_roll:.4e}, pitch={expected_pitch:.4e}, yaw={expected_yaw:.4e}\n")

    # 3) Hover is an equilibrium
    x0, u0 = hover_equilibrium(p, yaw=0.0)
    fx = f_dynamics(x0, u0, p, u_mode="Ttau")
    print("f(x0,u0) at hover:\n", fx)
    assert np.linalg.norm(fx) < 1e-9, "Hover should be a fixed point (‖f‖ < 1e-9)."
    print("Hover equilibrium passed.\n")

    # 4) Linearization sanity
    A, B, _, _ = linearize_hover(p, yaw=0.0)
    g = p.g
    G_block = A[3:6, 6:9]
    print("∂v̇/∂(φ,θ,ψ) block from A:\n", G_block)
    target = np.array([[0.0, -g, 0.0],
                       [ g ,  0.0, 0.0],
                       [0.0,  0.0, 0.0]])
    assert np.allclose(G_block, target, atol=1e-4, rtol=0), "Gravity cross-coupling block mismatch."
    print("Gravity cross-coupling signs/magnitudes passed.\n")

    # 5) Gains: ∂v̇z/∂T = -1/m ; ∂ω̇/∂τ = J^{-1}
    dvz_dT = B[5, 0]
    print("B[5,0] = ∂v̇z/∂T =", dvz_dT, " expected ≈", -1.0/p.m)
    assert np.isclose(dvz_dT, -1.0/p.m, atol=1e-6, rtol=0)
    Jinv = np.linalg.inv(p.J)
    rot_block = B[9:12, 1:4]
    print("Rotational block Bωτ (should be J^{-1}):\n", rot_block)
    assert np.allclose(rot_block, Jinv, atol=1e-8, rtol=0), "Rotational gains must equal J^{-1}."
    print("Rotational gains passed.\n")

    # 6) Allocation forward/backward round-trip
    rng = np.random.default_rng(2)
    omega2 = (rng.random(4) * (p.max_rpm**2))
    Ttau = thrust_torques_from_omegas2(omega2, p)
    omega2_rt = omegas2_from_thrust_torques(Ttau, p)
    print("Round-trip ‖ω² - (M_inv M) ω²‖ =", np.linalg.norm(omega2 - omega2_rt))
    assert np.allclose(omega2, omega2_rt, atol=1e-6, rtol=1e-9), "Forward/inverse allocation round-trip failed."
    print("Forward/inverse round-trip passed.\n")

    print("All tests passed")

if __name__ == "__main__":
    main()
