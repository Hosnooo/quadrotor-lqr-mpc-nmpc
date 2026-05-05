#!/usr/bin/env python3
from __future__ import annotations
"""
plot_nmpc_sweepN_summary.py

"""
import os, glob, math
import numpy as np
import matplotlib.pyplot as plt

from macros import SIM_DIR
from plotting import apply_style, save_fig, state_labels

RESULTS_SUBDIR = "NMPC_sweepN_dt020"

def _load_cases(results_dir: str):
    files = sorted(glob.glob(os.path.join(results_dir, "nmpcN*_dt*.npz")))
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
            "J": float(z["J"]),
            "sat_frac": float(z["sat_frac"]),
            "idxs": z["idxs"].astype(int),
            "ts": z["ts_mat"] if "ts_mat" in z.files else z["ts_settle"].astype(float),
            "os": z["os_mat"] if "os_mat" in z.files else z["os_pct"].astype(float),
            "T": z["T"],
            "E": z["E"],
            "O": z["O"] if "O" in z.files else None,
        })
    return sorted(data, key=lambda d: d["N"])

def _peak_angles_rad(E: np.ndarray) -> tuple[float, float, float]:
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

    Ns       = np.array([c["N"] for c in cases], dtype=int)
    J_all    = np.array([c["J"] for c in cases], dtype=float)
    sat_frac = np.array([c["sat_frac"] for c in cases], dtype=float)

    idxs = cases[0]["idxs"].astype(int)
    labs = state_labels()
    idx_to_label = {int(i): labs[int(i)] for i in idxs}

    ts_mat = np.vstack([c["ts"] for c in cases]).astype(float)
    os_mat = np.vstack([c["os"] for c in cases]).astype(float)

    peaks_phi = np.array([_peak_angles_rad(c["E"])[0] for c in cases], dtype=float)
    peaks_tht = np.array([_peak_angles_rad(c["E"])[1] for c in cases], dtype=float)
    peaks_psi = np.array([_peak_angles_rad(c["E"])[2] for c in cases], dtype=float)

    rpm_avg = np.array([np.mean(c["O"]) if c["O"] is not None else np.nan for c in cases], dtype=float)

    # ----------- Summary metrics (2x3) -----------
    fig_m, axs = plt.subplots(2,3, figsize=(14.5, 8.5))
    axJ, axTs, axOs = axs[0,0], axs[0,1], axs[0,2]
    axPkAng, axSat, axAvg = axs[1,0], axs[1,1], axs[1,2]

    axJ.plot(Ns, J_all, marker="o", linewidth=2.0)
    axJ.set_xlabel("Horizon N"); axJ.set_ylabel("Continuous‑time cost J")
    axJ.grid(True, alpha=0.3)

    for j, i in enumerate(idxs):
        axTs.plot(Ns, ts_mat[:, j], marker="o", linewidth=2.0, label=idx_to_label[int(i)])
    axTs.set_xlabel("Horizon N"); axTs.set_ylabel(r"Settling time [s] (2\% band)")
    axTs.grid(True, alpha=0.3)
    axTs.legend(loc='upper right', fontsize=15)

    for j, i in enumerate(idxs):
        if int(i) == 8:  # drop yaw from overshoot panel
            continue
        axOs.plot(Ns, os_mat[:, j], marker="o", linewidth=2.0, label=idx_to_label[int(i)])
    axOs.set_xlabel("Horizon N"); axOs.set_ylabel(r"Overshoot [\%]")
    axOs.grid(True, alpha=0.3)
    axOs.legend(loc='upper right', fontsize=15)

    axPkAng.plot(Ns, peaks_phi, marker="o", linewidth=2.0, label=r"$\phi$ (roll)")
    axPkAng.plot(Ns, peaks_tht, marker="o", linewidth=2.0, label=r"$\theta$ (pitch)")
    axPkAng.plot(Ns, peaks_psi, marker="o", linewidth=2.0, label=r"$\psi$ (yaw)")
    axPkAng.set_xlabel("Horizon N"); axPkAng.set_ylabel("Peak angle [rad]")
    axPkAng.grid(True, alpha=0.3)
    axPkAng.legend(loc='upper right', fontsize=15)

    axSat.plot(Ns, 100.0*sat_frac, marker="o", linewidth=2.0)
    axSat.set_xlabel("Horizon N"); axSat.set_ylabel(r"Saturation [\% of steps]")
    axSat.grid(True, alpha=0.3)

    axAvg.plot(Ns, rpm_avg, marker="o", linewidth=2.0)
    axAvg.set_xlabel("Horizon N"); axAvg.set_ylabel("Average rotor speed [RPM]")
    axAvg.grid(True, alpha=0.3)

    summary_pdf = os.path.join(results_dir, "nmpc_sweepN_summary_metrics.pdf")
    save_fig(fig_m, summary_pdf, tight=True)
    print(f"saved -> {summary_pdf}")

    # ----------- Error overlays (2x2) -----------
    fig_e, axs_e = plt.subplots(2,2, figsize=(12.0, 8.0))
    ax_px, ax_py, ax_pz, ax_yaw = axs_e[0,0], axs_e[0,1], axs_e[1,0], axs_e[1,1]
    ax_map = {idxs[0]: ax_px, idxs[1]: ax_py, idxs[2]: ax_pz, idxs[3]: ax_yaw}

    for case in cases:
        T = case["T"]; E = case["E"]; N_case = case["N"]
        for j, i in enumerate(idxs):
            ax = ax_map[int(i)]
            ax.plot(T, E[:, int(i)], linewidth=1.5, label=f"N={N_case}")

    ax_px.set_xlabel("t [s]"); ax_px.set_ylabel(r"$|p_x - p_{x,r}|$ [m]"); ax_px.grid(True, alpha=0.3)
    ax_py.set_xlabel("t [s]"); ax_py.set_ylabel(r"$|p_y - p_{y,r}|$ [m]"); ax_py.grid(True, alpha=0.3)
    ax_pz.set_xlabel("t [s]"); ax_pz.set_ylabel(r"$|p_z - p_{z,r}|$ [m]"); ax_pz.grid(True, alpha=0.3)
    ax_yaw.set_xlabel("t [s]"); ax_yaw.set_ylabel(r"$|\psi - \psi_r|$ [rad]"); ax_yaw.grid(True, alpha=0.3)

    handles, labels = ax_px.get_legend_handles_labels()
    n_cols = len(labels)
    fig_e.legend(handles, labels,
                 loc='upper center', bbox_to_anchor=(0.5, -0.03),
                 ncol=n_cols, frameon=False,
                 columnspacing=1.2, handlelength=2.0, labelspacing=0.7, fontsize=22)
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    errors_pdf = os.path.join(results_dir, "nmpc_sweepN_errors_timeseries.pdf")
    save_fig(fig_e, errors_pdf, tight=True)
    print(f"saved -> {errors_pdf}")

if __name__ == "__main__":
    main()
