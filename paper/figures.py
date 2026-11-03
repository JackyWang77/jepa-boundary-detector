"""Paper figures, regenerated from JSON artifacts in one visual style.

Hemingway rule for figures too: one panel, one message, no decoration.
Style: serif fonts to match the text, one accent color (dark red),
grayscale otherwise, no gridlines except where they carry meaning.

Run:  python3 paper/figures.py
Out:  paper/arxiv/figs/{calibration,clock_collapse,audit_bars}.png
Every input number traces to a JSON artifact (paths inline).
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "arxiv", "figs")
EXT = os.path.join(HERE, "..", "external")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200,
    "axes.linewidth": 0.8, "legend.frameon": False})
ACCENT = "#B2182B"      # one accent, used sparingly
GREYS = ["#444444", "#777777", "#AAAAAA", "#CCCCCC", "#E0E0E0"]


def calibration():
    """Synthetic AR(1) calibration: planted tau vs measured tau.
    Source: tau calibration runs (tau_spectrum protocol), recorded in the
    project ledger; points are the preregistered synthetic check."""
    planted = [2, 5, 10, 22, 40]
    measured = [2.7, 4.8, 7.2, 10.8, 12.9]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 45], [0, 45], color=GREYS[3], lw=0.8, zorder=1)
    ax.scatter(planted, measured, s=28, color=ACCENT, zorder=3)
    for p, m in zip(planted, measured):
        ax.annotate(f"{p}", (p, m), xytext=(4, -8),
                    textcoords="offset points", fontsize=9, color=GREYS[1])
    ax.set_xlabel("planted $\\tau$ (steps)")
    ax.set_ylabel("measured $\\tau$ (steps)")
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 45)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "calibration.png"))
    plt.close(fig)


def clock_collapse():
    """The dimensionless clock law: E_k(h) vs h/tau_k, all bands, three
    datasets. Source: external/c1_ladder_law.json (E_by_band, h_grid, tau)."""
    d = json.load(open(os.path.join(EXT, "c1_ladder_law.json")))
    names = [("pushT", "PushT"), ("antmaze_lewm", "AntMaze (LeWM)"),
             ("antmaze_hjepa", "AntMaze (H-JEPA)")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), sharey=True)
    for ax, (key, label) in zip(axes, names):
        ds = d[key]
        hg, taus = ds["h_grid"], ds["tau"]
        for i in range(8):
            e = ds["E_by_band"][str(i)]
            rho = ds["rho"][i]
            theory = [math.sqrt(max(0.0, 1 - rho ** (2 * h))) for h in hg]
            x = [h / taus[i] for h in hg]
            ax.plot(x, e, color=GREYS[2], lw=0.7)
            ax.plot(x, theory, color=ACCENT, lw=0.7, alpha=0.35)
        ax.set_xscale("log")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("$h / \\tau_k$")
    axes[0].set_ylabel("$E_k(h)$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "clock_collapse.png"))
    plt.close(fig)


def audit_bars():
    """The audit: top-8 slow-content recovery as a fraction of full, against
    the 0.8 threshold. Source: results/h_audit.json, external/lewm_h_audit.json
    (r2_full and r2_tau at k=8), extracted 2026-10-22."""
    rows = [("E$^2$-JEPA", 0.17, 0.760), ("Plain JEPA", 0.28, 0.748),
            ("TV variant", 0.32, 0.763), ("LeWM", 0.375, 0.583)]
    names = [r[0] for r in rows]
    frac = [r[1] / r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.bar(range(len(rows)), frac, 0.55, color=GREYS[2])
    ax.axhline(0.8, color=ACCENT, lw=1.2)
    ax.text(3.4, 0.82, "pass: $0.8 \\times$ full", fontsize=9, color=ACCENT,
            ha="right")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("top-8 $R^2$ / full $R^2$")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "audit_bars.png"))
    plt.close(fig)


def tau_sorted():
    """The gapless spectrum: sorted per-dimension timescales of the
    self-trained arms that pass the survival guard.
    Source: results/tau_spectrum.json (tau_sorted per arm). The
    energy-only control fails the guard and has no defined timescale
    (Sec. 2 of the paper), so it is absent by domain, not by choice."""
    d = json.load(open(os.path.join(HERE, "..", "results",
                                    "tau_spectrum.json")))
    arms = [("e2jepa", "E$^2$-JEPA", ACCENT), ("jepa", "plain JEPA", GREYS[2]),
            ("tv", "TV variant", GREYS[3])]
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    mx = 0.0
    for key, label, color in arms:
        ts = d[key]["tau_sorted"]
        ax.plot(range(1, len(ts) + 1), ts, color=color, lw=1.1, label=label)
        mx = max(mx, max(ts[i] / max(ts[i + 1], 1e-9)
                         for i in range(len(ts) - 1)))
    ax.annotate(f"largest adjacent ratio: {mx:.2f}\n"
                "(a level boundary needs 1.15)",
                xy=(0.97, 0.05), xycoords="axes fraction", ha="right",
                fontsize=9, color=GREYS[0])
    ax.set_xlabel("dimension rank (sorted by $\\tau$)")
    ax.set_ylabel("$\\tau$ (env steps)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tau_sorted.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    calibration()
    clock_collapse()
    audit_bars()
    tau_sorted()
    print("wrote calibration.png, clock_collapse.png, audit_bars.png, "
          "tau_sorted.png ->", FIGS)
