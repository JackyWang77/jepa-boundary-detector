"""Generate proposal figures from real measurement results.

Inputs : results/tau_spectrum.json (five self-trained regimes)
         external/lewm_tau.json   (official LeWM world model)
Output : proposal/figs/fig{1..5}.{pdf,png}

fig1  universal spectral gap collapse        (measured)
fig2  A/B dissociation + neutral-mode probe  (measured)
fig3  two dead ends: flat half-spectrum probes(measured)
fig4  pseudo-slow tau: existence precondition(measured)
fig5  thermodynamic mapping + T_k mechanism  (schematic)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

T = json.load(open(os.path.join(ROOT, "results", "tau_spectrum.json")))
L = json.load(open(os.path.join(ROOT, "external", "lewm_tau.json")))

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 8,
    "axes.titlesize": 9.5, "figure.dpi": 150, "savefig.bbox": "tight",
})
C = {"jepa": "#4477AA", "e2jepa": "#EE6677", "bc": "#228833",
     "bc_flow": "#CCBB44", "lewm": "#AA3377", "energy_only": "#66CCEE",
     "tv": "#8B0000"}
NICE = {"jepa": "JEPA", "e2jepa": "E$^2$-JEPA", "bc": "BC",
        "bc_flow": "BC-flow", "lewm": "LeWM", "energy_only": "energy-only",
        "tv": "E$^2$-JEPA + $\\boldsymbol{T}_k$"}


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1 ----
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
ax = axes[0]
for arm in ["jepa", "e2jepa", "bc", "bc_flow", "tv"]:
    ts = np.asarray(T[arm]["tau_sorted"])
    ax.plot(np.arange(1, len(ts) + 1), ts, color=C[arm], lw=1.4,
            label=f"{NICE[arm]}  (gap $\\times${T[arm]['tau_gap']['ratio']:.3f})")
ax.set_yscale("log")
ax.set_xlabel("latent dimension (sorted by $\\tau$)")
ax.set_ylabel("$\\tau$ [env steps]")
ax.set_title("(a) self-trained regimes (PushT)")
ax.legend(loc="upper right")
ax.set_xlim(0, 130)

ax = axes[1]
ts = np.asarray(L["tau_env"]["tau_sorted"])
ax.plot(np.arange(1, len(ts) + 1), ts, color=C["lewm"], lw=1.6)
ax.set_yscale("log")
ax.set_xlabel("latent dimension (sorted by $\\tau$, 192-d)")
ax.set_title(f"(b) official LeWM model  (gap $\\times$"
             f"{L['tau_env']['tau_gap']['ratio']:.3f})")
ax.set_xlim(0, 194)
ax.annotate("no hierarchy boundary\nanywhere in the spectrum",
            xy=(0.72, 0.82), xycoords="axes fraction", ha="center",
            fontsize=8.5, color="0.25")
fig.tight_layout()
save(fig, "fig1")

# ---------------------------------------------------------------- fig 2 ----
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1))
ax = axes[0]
ts = np.asarray(T["e2jepa"]["tau_sorted"])
ax.fill_between(np.arange(1, len(ts) + 1), ts.min(), ts.max(), alpha=0.18,
                color=C["e2jepa"])
ax.plot(np.arange(1, len(ts) + 1), ts, color=C["e2jepa"], lw=1.4)
ax.set_ylim(7, 11.5)
ax.set_title("(a) estimator A (data side)\nE$^2$-JEPA latents: $\\tau = 9\\pm1$")
ax.set_xlabel("latent dimension")

ax = axes[1]
ts = np.sort(np.asarray(T["e2jepa"]["jac"]["tau_rank"]))[::-1]
ax.plot(np.arange(1, len(ts) + 1), ts, color=C["e2jepa"], lw=1.4)
ax.set_yscale("log")
ax.set_title("(b) estimator B (model side)\nsame latents: $\\tau$ spans $10^{-1}$--$10^{1}$")
ax.set_xlabel("Jacobian eigen-rank")

ax = axes[2]
ts = np.sort(np.asarray(L["jac"]["tau_rank"]))[::-1]
r = np.arange(1, len(ts) + 1)
ax.plot(r, ts, color=C["lewm"], lw=1.4)
ax.plot(r[ts >= 1000], ts[ts >= 1000], "o", ms=3.5, color="#8B0000")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("(c) LeWM model side\n7 neutral modes $=$ dim of physical state")
ax.set_xlabel("Jacobian eigen-rank")
ax.annotate("7 modes with $|\\lambda|\\geq 1$\nat every context",
            xy=(4, 700), fontsize=8, color="#8B0000", ha="center")
axes[0].set_ylabel("$\\tau$ [env steps]")
axes[1].set_ylabel("$\\tau$ [env steps]")
axes[2].set_ylabel("$\\tau$ [macro steps]")
fig.tight_layout()
save(fig, "fig2")

# ---------------------------------------------------------------- fig 3 ----
fig, ax = plt.subplots(figsize=(6.6, 3.4))
models = ["e2jepa", "lewm"]
labels = ["block pose", "agent pos"]
width, x0 = 0.19, np.array([0.0, 1.0])
for j, (arm, half) in enumerate([("e2jepa", "slow"), ("e2jepa", "fast"),
                                 ("lewm", "slow"), ("lewm", "fast")]):
    src = T[arm] if arm != "lewm" else L
    vals = [src["probes"][half]["block"], src["probes"][half]["agent"]]
    pos = x0 + (j - 1.5) * width
    shade = 0.45 if half == "fast" else 1.0
    col = matplotlib.colors.to_rgb(C[arm])
    col = tuple(min(1, c * shade + (1 - shade)) for c in col)
    bars = ax.bar(pos, vals, width * 0.92, color=col,
                  label=f"{NICE[arm]} {'slow' if half == 'slow' else 'fast'} half")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", fontsize=7.2)
ax.set_xticks(x0)
ax.set_xticklabels(labels)
ax.set_ylabel("linear probe $R^2$")
ax.set_ylim(0, 0.78)
ax.legend(ncol=2, loc="upper left", fontsize=7.5)
ax.text(0.5, 0.955, "E$^2$-JEPA: PR $=114.5$ (overcomplete)",
        transform=ax.transAxes, ha="center", fontsize=8, color=C["e2jepa"])
ax.text(0.5, 0.885, "LeWM: PR $=6.2$ (compressed)",
        transform=ax.transAxes, ha="center", fontsize=8, color=C["lewm"])
fig.tight_layout()
save(fig, "fig3")

# ---------------------------------------------------------------- fig 4 ----
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
ax = axes[0]
for arm in ["energy_only", "jepa"]:
    ts = np.asarray(T[arm]["tau_sorted"])
    ax.plot(np.arange(1, len(ts) + 1), ts, color=C[arm], lw=1.4,
            label=f"{NICE[arm]}  (median {np.median(ts):.1f})")
ax.set_yscale("log")
ax.set_xlabel("latent dimension (sorted by $\\tau$)")
ax.set_ylabel("$\\tau$ [env steps]")
ax.set_title("(a) collapsed latents measure slow $\\tau$ -- pseudo-slow")
ax.legend()

ax = axes[1]
arms = ["energy_only", "jepa"]
pos = np.arange(2)
VAR_COL = {"block": "#4477AA", "agent": "0.55"}
for i, arm in enumerate(arms):
    for j, var in enumerate(["block", "agent"]):
        v = T[arm]["probes"]["full"][var]
        ax.bar(i + (j - 0.5) * 0.34, max(v, 0.0), 0.32,
               color=VAR_COL[var],
               label=var if i == 0 else None)
        ax.text(i + (j - 0.5) * 0.34, max(v, 0) + 0.015, f"{v:.2f}",
                ha="center", fontsize=7.5)
ax.set_xticks(pos)
ax.set_xticklabels(["energy-only\nPR $=4.6$, dim std $=0.003$",
                    "JEPA\nPR $=57.2$"])
ax.set_ylabel("linear probe $R^2$")
ax.set_title("(b) ...but carries almost no task information")
ax.set_ylim(0, 0.9)
ax.legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
save(fig, "fig4")

# ---------------------------------------------------------------- fig 5 ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.5),
                               gridspec_kw={"width_ratios": [1.15, 1]})
axL.set_xlim(0, 10)
axL.set_ylim(0, 10)
axL.axis("off")


def box(x, y, w, h, text, fc):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       fc=fc, ec="0.3", lw=0.8)
    axL.add_patch(b)
    axL.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


box(0.3, 7.3, 3.6, 1.7, "$U$: prediction error\n(internal energy)", "#fde0dd")
box(0.3, 4.1, 3.6, 1.7, "$S_k$: band entropy\n$\\frac{1}{2}\\log\\det\\Sigma_k$", "#d9f2d9")
axL.text(5.05, 6.55, "$-$", fontsize=17, ha="center", va="center")
axL.add_patch(FancyArrowPatch((3.95, 8.1), (6.4, 7.6),
                              arrowstyle="-|>", mutation_scale=12, lw=0.9))
axL.add_patch(FancyArrowPatch((3.95, 4.9), (6.4, 5.4),
                              arrowstyle="-|>", mutation_scale=12, lw=0.9))
box(6.5, 5.4, 3.3, 2.6, "$F = U - \\sum_k T_k\\, S_k$\n(free energy)", "#e6e6fa")
# thermometer: temperature gradient across tau-ranked bands
for i in range(4):
    heat = 1 - i / 3.0
    col = (0.92 - 0.55 * (1 - heat), 0.45 + 0.2 * heat, 0.35 + 0.45 * heat)
    b = FancyBboxPatch((7.0 + i * 0.72, 3.2), 0.62, 1.1,
                       boxstyle="round,pad=0.06", fc=col, ec="0.3", lw=0.6)
    axL.add_patch(b)
    axL.text(7.31 + i * 0.72, 2.75, f"$T_{i+1}$", ha="center", fontsize=8)
axL.text(5.0, 2.0, "temperature gradient across $\\tau$-ranked bands:\n"
                   "high $T$ buys entropy in slow bands, low $T$ refines fast",
         ha="center", va="top", fontsize=8)

ts_m = np.asarray(T["e2jepa"]["tau_sorted"])
axR.plot(np.arange(1, 129), ts_m, color="0.45", lw=1.6,
         label="scalar $T$ (measured): gap $\\times$1.01")
ts_tv = np.asarray(T["tv"]["tau_sorted"])
axR.plot(np.arange(1, 129), ts_tv, color="#8B0000", lw=1.6,
         label="vector $\\boldsymbol{T}_k$ (measured): gap $\\times$1.02")
axR.set_yscale("log")
axR.set_xlabel("latent dimension (sorted by $\\tau$)")
axR.set_ylabel("$\\tau$ [env steps]")
axR.legend(loc="upper right", fontsize=7.5)
axR.set_title("measured: the gap does not open")
fig.tight_layout()
save(fig, "fig5")

print("all figures done")
