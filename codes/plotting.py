#!/usr/bin/env python3
from __future__ import annotations
"""
Reusable plotting style + helpers for this project.

apply_style(theme="latex", context="paper", font_scale=1.0)
state_labels(), input_labels()
label_state_axis(ax, axis="x"), label_input_axis(ax, axis="y")
heatmap(ax, M, ...), save_fig(fig, path)
"""

from typing import Iterable, Optional, Sequence
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from dynamics import QuadParams
__all__ = [
    "apply_style", "state_labels", "input_labels",
    "label_state_axis", "label_input_axis",
    "heatmap", "save_fig",
]

# Labels
def state_labels() -> list[str]:
    return [r"$p_x$", r"$p_y$", r"$p_z$",
            r"$v_x$", r"$v_y$", r"$v_z$",
            r"$\phi$", r"$\theta$", r"$\psi$",
            r"$p$", r"$q$", r"$r$"]

def state_labels_ref() -> list[str]:
    return [r"$p_{x,ref}$", r"$p_{y,ref}$", r"$p_{z,ref}$",
            r"$\phi_{ref}$", r"$\theta_{ref}$", r"$\psi_{ref}$"]

def input_labels() -> list[str]:
    return [r"$\delta T$", r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]

# Style
def _sizes_for(context: str, font_scale: float = 1.0):
    base = 10.0 * font_scale
    if context not in {"paper", "notebook"}:
        context = "paper"
    if context == "notebook":
        base *= 1.2
    return {
        "axes.labelsize": base,
        "axes.titlesize": base * 1.1,
        "legend.fontsize": base * 0.9,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
    }

def _latex_available() -> bool:
    has_latex = shutil.which("latex") is not None
    has_dvi   = shutil.which("dvipng") is not None or shutil.which("dvisvgm") is not None
    return bool(has_latex and has_dvi)

def apply_style(theme: str = "latex", context: str = "paper",
                font_scale: float = 1.0, preset: str | None = None):
    """
    Apply global plotting style.
    - theme: "latex" or "default"
    - context: ignored here but kept for API compatibility
    - font_scale: multiplies all base font sizes
    - preset: if "pub", set publication-ready rcParams (larger fonts, thicker lines)
    """
    
    if theme == "latex":
        mpl.rcParams.update({
            "text.usetex": True,         # keep False unless you have LaTeX runtime
            "font.family": "serif",
            "mathtext.fontset": "stix",
        })
    else:
        mpl.rcdefaults()

    if preset == "pub":
        s = font_scale
        mpl.rcParams.update({
            # font sizes
            "font.size":            20 * s,
            "axes.labelsize":       15 * s,
            "axes.titlesize":       20 * s,
            "xtick.labelsize":      20 * s,
            "ytick.labelsize":      20 * s,
            "legend.fontsize":      25 * s,

            # line/marker widths
            "lines.linewidth":      2.0,
            "axes.linewidth":       1.0,

            # grid
            "axes.grid":            True,
            "grid.alpha":           0.25,

            # ticks
            "xtick.direction":      "in",
            "ytick.direction":      "in",
            "xtick.major.size":     5.0,
            "xtick.major.width":    1.0,
            "ytick.major.size":     5.0,
            "ytick.major.width":    1.0,

            # legends
            "legend.frameon":       True,
            "legend.fancybox":      True,
            "legend.borderaxespad": 0.6,

            # output
            "savefig.dpi":          300,
            "figure.dpi":           140,
            "pdf.fonttype":         42,
            "ps.fonttype":          42,
        })

# Axis labeling
def label_state_axis(ax: mpl.axes.Axes, axis: str = "x", rotate: int = 45, ha: str = "right"):
    labs = state_labels()
    reflabs = state_labels_ref()
    idx = range(len(labs))
    if axis == "x":
        ax.set_xticks(idx); ax.set_xticklabels(labs, rotation=rotate, ha=ha)
    else:
        ax.set_yticks(idx); ax.set_yticklabels(labs, rotation=0, ha="right")

def label_input_axis(ax: mpl.axes.Axes, axis: str = "y"):
    labs = input_labels()
    idx = range(len(labs))
    if axis == "x":
        ax.set_xticks(idx); ax.set_xticklabels(labs, rotation=0, ha="center")
    else:
        ax.set_yticks(idx); ax.set_yticklabels(labs, rotation=0, ha="right")

# Heatmap / Saving
def heatmap(ax: mpl.axes.Axes, M: np.ndarray,
            xlabels: Optional[Sequence[str]] = None,
            ylabels: Optional[Sequence[str]] = None,
            title: Optional[str] = None,
            cmap: str = "viridis",
            show_colorbar: bool = True,
            extent: Optional[Iterable[float]] = None,
            origin: str = "upper"):
    """Generic heatmap with optional axis labels (strings) and colorbar."""
    im = ax.imshow(M, aspect="auto", cmap=cmap, extent=extent, origin=origin)
    im.set_rasterized(True)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=45, ha="right")
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    if title:
        ax.set_title(title)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

def save_fig(fig, path, tight=True):
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, bbox_inches="tight")

import numpy as _np
import matplotlib.pyplot as _plt

def plot_states_stack(T, X, xr, save_path, legend_pos=("center right", (1.0, 0.35)), fontsize=10):
    """
    Four stacked panels: position, velocity, angles, rates.
    Shows dotted refs for (px,py,pz,psi).
    """
    s_labs = state_labels()
    s_labs_ref = state_labels_ref()
    fig, axs = _plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    # positions
    axs[0].plot(T, X[:,0], label=fr"{s_labs[0]}")
    axs[0].plot(T, X[:,1], label=fr"{s_labs[1]}")
    axs[0].plot(T, X[:,2], label=fr"{s_labs[2]}")
    axs[0].plot(T, _np.full_like(T, xr[0]), ":", label=fr"{s_labs_ref[0]}")
    axs[0].plot(T, _np.full_like(T, xr[1]), ":", label=fr"{s_labs_ref[1]}")
    axs[0].plot(T, _np.full_like(T, xr[2]), ":", label=fr"{s_labs_ref[2]}")
    axs[0].set_ylabel("position [m]")
    loc, anchor = legend_pos
    axs[0].legend(ncol=2, loc=loc, bbox_to_anchor=anchor, fontsize=fontsize)

    # velocities
    for j in (3,4,5): axs[1].plot(T, X[:,j], label=fr"{s_labs[j]}")
    axs[1].set_ylabel("velocity [m/s]")
    axs[1].legend(ncol=1, loc="center right", fontsize=fontsize)

    # angles
    for j in (6,7,8): axs[2].plot(T, X[:,j], label=fr"{s_labs[j]}")
    axs[2].plot(T, _np.full_like(T, xr[8]), ":", label=fr"{s_labs_ref[3]}")
    axs[2].set_ylabel("angles [rad]")
    axs[2].legend(ncol=1, loc="center right", fontsize=fontsize)

    # rates
    for j in (9,10,11): axs[3].plot(T, X[:,j], label=fr"{s_labs[j]}")
    axs[3].set_ylabel("rates [rad/s]"); axs[3].set_xlabel("time [s]")
    axs[3].legend(ncol=1, loc="center right", fontsize=fontsize)

    save_fig(fig, save_path); _plt.close(fig)

def plot_rpms_with_sats(T, O, params, save_path, legend=("center right", (1.0, 0.3)), fontsize=11):
    """
    Motor RPMs with dashed min/max RPM lines.
    O: (N,4) with Ω1..Ω4 in RPM.
    """
    rpm_min = float(params.min_rpm); rpm_max = float(params.max_rpm)
    fig, ax = _plt.subplots(figsize=(10, 5))
    motor_labels = [r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"]
    for j in range(O.shape[1]):
        ax.plot(T, O[:, j], label=fr"{motor_labels[j]}", linewidth=1.4)
    ax.axhline(rpm_max, linestyle="--", linewidth=1.0)
    ax.axhline(rpm_min, linestyle="--", linewidth=1.0)
    ax.set_xlabel("time [s]"); ax.set_ylabel("motor speed [RPM]")
    loc, anchor = legend
    ax.legend(ncol=1, loc=loc, bbox_to_anchor=anchor, fontsize=fontsize)
    save_fig(fig, save_path); _plt.close(fig)

def _legend_outside_right(ax, ncol=1, fontsize=12, anchor_x=1.02, anchor_y=0.5):
    """Place legend outside the axes to the right using bbox_to_anchor."""
    leg = ax.legend(
        ncol=ncol, loc="center left", bbox_to_anchor=(anchor_x, anchor_y),
        fontsize=fontsize, frameon=True
    )
    return leg

def _plot_states_2x2(T, X, xr, out_path: str, lw: float = 2.0,
                     legend_fs: int = 12, title_fs: int = 14):
    """
    Four panels in a 2×2 grid:
      [pos] [vel]
      [angles] [rates]
    Puts legends outside (right) using bbox_to_anchor.
    """
    s_labs = state_labels()
    s_labs_ref = state_labels_ref()
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 7.8), sharex=True)
    ax00, ax01 = axs[0, 0], axs[0, 1]
    ax10, ax11 = axs[1, 0], axs[1, 1]

    # positions
    ax = ax00
    ax.plot(T, X[:, 0], label=rf"{s_labs[0]}", linewidth=lw)
    ax.plot(T, X[:, 1], label=rf"{s_labs[1]}", linewidth=lw)
    ax.plot(T, X[:, 2], label=rf"{s_labs[2]}", linewidth=lw)
    ax.plot(T, np.full_like(T, xr[0]), ":", label=rf"{s_labs_ref[0]}", linewidth=lw-0.6)
    ax.plot(T, np.full_like(T, xr[1]), ":", label=rf"{s_labs_ref[1]}", linewidth=lw-0.6)
    ax.plot(T, np.full_like(T, xr[2]), ":", label=rf"{s_labs_ref[2]}", linewidth=lw-0.6)
    ax.set_title("position [m]", fontsize=title_fs)
    _legend_outside_right(ax, ncol=1, fontsize=legend_fs)
    ax.grid(True, alpha=0.25)

    # velocities
    ax = ax01
    for j in (3, 4, 5):
        ax.plot(T, X[:, j], label=rf"{s_labs[j]}", linewidth=lw)
    ax.set_title("velocity [m/s]", fontsize=title_fs)
    _legend_outside_right(ax, ncol=1, fontsize=legend_fs)
    ax.grid(True, alpha=0.25)

    # angles
    ax = ax10
    for j in (6, 7, 8):
        ax.plot(T, X[:, j], label=rf"{s_labs[j]}", linewidth=lw)
    ax.plot(T, np.full_like(T, xr[6]), ":", label=rf"{s_labs_ref[3]}", linewidth=lw-0.6)
    ax.plot(T, np.full_like(T, xr[7]), ":", label=rf"{s_labs_ref[4]}", linewidth=lw-0.6)
    ax.plot(T, np.full_like(T, xr[8]), ":", label=rf"{s_labs_ref[5]}", linewidth=lw-0.6)
    ax.set_title("angles [rad]", fontsize=title_fs)
    _legend_outside_right(ax, ncol=1, fontsize=legend_fs)
    ax.grid(True, alpha=0.25)

    # rates
    ax = ax11
    for j in (9, 10, 11):
        ax.plot(T, X[:, j], label=rf"{s_labs[j]}", linewidth=lw)
    ax.set_title("rates [rad/s]", fontsize=title_fs)
    _legend_outside_right(ax, ncol=1, fontsize=legend_fs)
    ax.grid(True, alpha=0.25)

    # common labels
    for ax in axs[1, :]:
        ax.set_xlabel("time [s]")
    for ax in axs[:, 0]:
        ax.set_ylabel("")

    # make room for outside legends on the right
    fig.subplots_adjust(right=0.78, wspace=0.25, hspace=0.25)
    save_fig(fig, out_path)
    plt.close(fig)

def _plot_rpms(T: np.ndarray, O: np.ndarray, params: QuadParams,
                             out_path: str, lw: float = 2.0,
                             legend_fs: int = 12):
    """
    Plots Ω_i(t) with dashed Ω_min/Ω_max. Legend placed below using bbox_to_anchor.
    """
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    motor_labels = [r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"]
    for j in range(O.shape[1]):
        ax.plot(T, O[:, j], label=motor_labels[j], linewidth=lw)
    # saturation lines
    if params.max_rpm > 0:
        ax.axhline(params.max_rpm, linestyle="--", linewidth=lw-1.0, label=r"$\Omega_{\max}$")
    if params.min_rpm > 0:
        ax.axhline(params.min_rpm, linestyle="--", linewidth=lw-1.0, label=r"$\Omega_{\min}$")

    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"motor speed $\Omega$ [RPM]")
    # Legend centered under the axes
    leg = ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                    fontsize=legend_fs, frameon=True)
    ax.grid(True, alpha=0.25)

    fig.subplots_adjust(bottom=0.28)
    save_fig(fig, out_path)
    plt.close(fig)

def _plot_costates_stack(T, lam_dyn, lam_hat, out_path: str, legend_fs: int = 10):
    """
    Two stacked panels:
      top:  IPOPT multipliers (λ from defects, i.e., λ_{k+1})
      bottom: backward-recursed λ̂
    """
    if lam_dyn is None or lam_hat is None or len(lam_dyn) == 0:
        return
    t = _np.asarray(T)
    t_k = t[:-1]
    lam_dyn = _np.asarray(lam_dyn)
    lam_hat = _np.asarray(lam_hat)
    nx = lam_dyn.shape[1]

    fig, axs = _plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    ax0, ax1 = axs[0], axs[1]
    for i in range(nx):
        ax0.plot(t_k, lam_dyn[:, i], label=fr"$\lambda_{i}$")
        ax1.plot(t_k, lam_hat[:, i], label=fr"$\hat\lambda_{i}$", linestyle="--")
    ax0.set_ylabel(r"IPOPT $\lambda$"); ax1.set_ylabel(r"backward $\hat\lambda$")
    ax1.set_xlabel("time [s]")
    _legend_outside_right(ax0, ncol=3, fontsize=legend_fs)
    _legend_outside_right(ax1, ncol=3, fontsize=legend_fs)
    save_fig(fig, out_path); _plt.close(fig)

def _plot_pmp_residuals_inf(T, ru, rp, out_path: str):
    """
    Infinity-norm residuals over the horizon:
      ||r_u||_∞  for stationarity wrt W2,
      ||r_p||_∞  for multiplier consistency.
    """
    if ru is None or rp is None or len(ru) == 0:
        return
    t = _np.asarray(T)[:-1]
    ru = _np.asarray(ru); rp = _np.asarray(rp)
    ru_n = _np.linalg.norm(ru, ord=_np.inf, axis=1)
    rp_n = _np.linalg.norm(rp, ord=_np.inf, axis=1)

    fig = _plt.figure(figsize=(10, 4))
    _plt.plot(t, ru_n, label=r"$\|r_u\|_\infty$")
    _plt.plot(t, rp_n, label=r"$\|r_p\|_\infty$", linestyle="--")
    _plt.xlabel("time [s]"); _plt.ylabel("residual ($\infty$-norm)")
    _plt.grid(True, alpha=0.25)
    _legend_outside_right(_plt.gca(), ncol=1, fontsize=10, anchor_x=1.05, anchor_y=0.5)
    save_fig(fig, out_path); _plt.close(fig)



def main():
    pass


if __name__ == "__main__":
    main()
