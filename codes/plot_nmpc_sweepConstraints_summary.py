#!/usr/bin/env python3
from __future__ import annotations
"""
plot_nmpc_sweepConstraints_summary.py

"""

import os, glob, math
import numpy as np
import matplotlib.pyplot as plt

from macros import SIM_DIR
from plotting import apply_style, save_fig, state_labels

RESULTS_SUBDIR = "NMPC_sweepConstraints_dt020_N080"

def _load_cases(results_dir: str):
    files = sorted(glob.glob(os.path.join(results_dir, "nmpcC_*.npz")))
    files.pop()
    data = []
    for fp in files:
        try:
            z = np.load(fp, allow_pickle=True)
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue
        status = str(z["status"])
        if not status.startswith("solved"):
            continue
        data.append({
            "path": fp,
            "status": status,
            "N": int(z["N"]),
            "rpm_min": float(z["rpm_min"]),
            "rpm_max": float(z["rpm_max"]),
            "scale": float(z["scale"]),
            "J": float(z["J"]),
            "sat_frac": float(z["sat_frac"]),
            "idxs": z["idxs"].astype(int),
            "ts": z["ts_settle"].astype(float),
            "os": z["os_pct"].astype(float),
            "T": z["T"],
            "E": z["E"],
            "O": z["O"] if "O" in z.files else None,
        })
    return sorted(data, key=lambda d: d["rpm_max"])

def _peak_angles_rad(E: np.ndarray) -> tuple[float, float, float]:
    # Peak absolute roll/pitch/yaw from error array E
    return (float(np.max(np.abs(E[:,6]))),
            float(np.max(np.abs(E[:,7]))),
            float(np.max(np.abs(E[:,8]))))

def main():
    apply_style(preset="pub")
    results_dir = os.path.join(SIM_DIR, RESULTS_SUBDIR)
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    cases = _load_cases(results_dir)
    if not cases:
        raise RuntimeError("No solved cases found. Run the runner first.")

    idxs = cases[0]["idxs"].astype(int)
    labs = state_labels()
    idx_to_label = {int(i): labs[int(i)] for i in idxs}

    rpm_max  = np.array([c["rpm_max"] for c in cases], dtype=float)
    J_all    = np.array([c["J"] for c in cases], dtype=float)
    sat_frac = np.array([c["sat_frac"] for c in cases], dtype=float)
    ts_mat   = np.vstack([c["ts"] for c in cases]).astype(float)
    os_mat   = np.vstack([c["os"] for c in cases]).astype(float)

    # Peak angles and average RPM
    peaks_phi = np.array([_peak_angles_rad(c["E"])[0] for c in cases], dtype=float)
    peaks_tht = np.array([_peak_angles_rad(c["E"])[1] for c in cases], dtype=float)
    peaks_psi = np.array([_peak_angles_rad(c["E"])[2] for c in cases], dtype=float)
    rpm_avg   = np.array([np.mean(c["O"]) if c["O"] is not None else np.nan for c in cases], dtype=float)

    # ----------- Summary metrics (2x3) -----------
    fig_m, axs = plt.subplots(2,3, figsize=(14.5, 8.5))
    axJ, axTs, axOs = axs[0,0], axs[0,1], axs[0,2]
    axPkAng, axSat, axAvg = axs[1,0], axs[1,1], axs[1,2]

    # J vs constraint (rpm_max)
    axJ.plot(rpm_max, J_all, marker="o", linewidth=2.0)
    axJ.set_xlabel("$\Omega_{\max}$ [RPM]"); axJ.set_ylabel("Continuous‑time cost J")
    axJ.grid(True, alpha=0.3)

    # Settling vs rpm_max (2% band)
    for j, i in enumerate(idxs):
        axTs.plot(rpm_max, ts_mat[:, j], marker="o", linewidth=2.0, label=idx_to_label[int(i)])
    axTs.set_xlabel("$\Omega_{\max}$ [RPM]"); axTs.set_ylabel(r"Settling time [s] (2\% band)")
    axTs.grid(True, alpha=0.3)
    axTs.legend(loc='upper right', fontsize=15)

    # Overshoot vs rpm_max (drop yaw from this panel), legend upper-right
    for j, i in enumerate(idxs):
        if int(i) == 8:
            continue
        axOs.plot(rpm_max, os_mat[:, j], marker="o", linewidth=2.0, label=idx_to_label[int(i)])
    axOs.set_xlabel("$\Omega_{\max}$ [RPM]"); axOs.set_ylabel(r"Overshoot [\%]")
    axOs.grid(True, alpha=0.3)
    axOs.legend(loc='upper right', fontsize=15)

    # Peak angles vs rpm_max (rad), legend upper-right
    axPkAng.plot(rpm_max, peaks_phi, marker="o", linewidth=2.0, label=r"$\phi$ (roll)")
    axPkAng.plot(rpm_max, peaks_tht, marker="o", linewidth=2.0, label=r"$\theta$ (pitch)")
    axPkAng.plot(rpm_max, peaks_psi, marker="o", linewidth=2.0, label=r"$\psi$ (yaw)")
    axPkAng.set_xlabel("$\Omega_{\max}$ [RPM]"); axPkAng.set_ylabel("Peak angle [rad]")
    axPkAng.grid(True, alpha=0.3)
    axPkAng.legend(loc='upper right', fontsize=15)

    # Saturation fraction vs rpm_max
    axSat.plot(rpm_max, 100.0*sat_frac, marker="o", linewidth=2.0)
    axSat.set_xlabel("$\Omega_{\max}$ [RPM]"); axSat.set_ylabel(r"Saturation [\% of steps]")
    axSat.grid(True, alpha=0.3)

    # Avg rotor speed vs rpm_max
    axAvg.plot(rpm_max, rpm_avg, marker="o", linewidth=2.0)
    axAvg.set_xlabel("$\Omega_{\max}$ [RPM]"); axAvg.set_ylabel("Average rotor speed [RPM]")
    axAvg.grid(True, alpha=0.3)

    summary_pdf = os.path.join(results_dir, "nmpc_constraints_summary_metrics.pdf")
    save_fig(fig_m, summary_pdf, tight=True)
    print(f"saved -> {summary_pdf}")

    # ----------- Error overlays (2x2) -----------
    fig_e, axs_e = plt.subplots(2,2, figsize=(12.0, 8.0))
    ax_px, ax_py, ax_pz, ax_yaw = axs_e[0,0], axs_e[0,1], axs_e[1,0], axs_e[1,1]

    for case in cases:
        T = case["T"]; E = case["E"]; label = "$\Omega_{\max}$="+f"{int(round(case['rpm_max']))}"
        ax_px.plot(T, E[:, 0], linewidth=1.4, label=label)
        ax_py.plot(T, E[:, 1], linewidth=1.4, label=label)
        ax_pz.plot(T, E[:, 2], linewidth=1.4, label=label)
        ax_yaw.plot(T, E[:, 8], linewidth=1.4, label=label)

    # Elegant labels (no subplot titles)
    ax_px.set_xlabel("t [s]"); ax_px.set_ylabel(r"$|p_x - p_{x,r}|$ [m]"); ax_px.grid(True, alpha=0.3)
    ax_py.set_xlabel("t [s]"); ax_py.set_ylabel(r"$|p_y - p_{y,r}|$ [m]"); ax_py.grid(True, alpha=0.3)
    ax_pz.set_xlabel("t [s]"); ax_pz.set_ylabel(r"$|p_z - p_{z,r}|$ [m]"); ax_pz.grid(True, alpha=0.3)
    ax_yaw.set_xlabel("t [s]"); ax_yaw.set_ylabel(r"$|\psi - \psi_r|$ [rad]"); ax_yaw.grid(True, alpha=0.3)

    # Figure-level legend outside (bottom), ncol = number of entries
    handles, labels = ax_px.get_legend_handles_labels()
    # n_cols = len(labels) if len(labels) > 0 else 1
    n_cols = 3
    fig_e.legend(handles, labels,
                 loc='upper center', bbox_to_anchor=(0.5, -0.03),
                 ncol=n_cols, frameon=False,
                 columnspacing=1.2, handlelength=2.0, labelspacing=0.7, fontsize=22)

    # Leave room at bottom for the legend
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    errors_pdf = os.path.join(results_dir, "nmpc_constraints_errors_timeseries.pdf")
    save_fig(fig_e, errors_pdf, tight=True)
    print(f"saved -> {errors_pdf}")

if __name__ == "__main__":
    main()
