#!/usr/bin/env python3
"""Generate Figure 1: flat planning vs boundary-guided segmentation.

Outputs figs/rollout_flat_vs_seg_{en,cn}.{pdf,png}
en/cn differ only in label language; geometry is identical.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

ACCENT = "#B2182B"
INK = "#1a1a1a"
GRAY = "#8a8a8a"
EDGE_A = "#9a9a9a"
EDGE_B = "#D89A92"
WEDGE_A = "#c7c7c7"
WEDGE_B = "#F2CBC6"

T_MAX = 20.0
BOUNDS = [5.0, 11.0]  # unequal on purpose: boundaries are physical events


def err_flat(t):
    return 1.75 * (np.asarray(t, float) / T_MAX) ** 1.6


SEGMENTS = [(0.0, 5.0), (5.0, 11.0), (11.0, 20.0)]


def seg_err(t, s0, s1):
    u = (np.asarray(t, float) - s0) / (s1 - s0)
    return 0.55 * np.clip(u, 0.0, 1.0) ** 1.6


L = {
    "en": dict(
        title_a="(a) Flat planning",
        title_b="(b) Boundary-guided segmentation",
        err="rollout error compounds",
        reset="re-anchor to real observation:\ndrift cleared",
        subgoal="Subgoal {}",
        detect="Detection {}",
        brace_a="one joint optimization, a$_{1}$–a$_{20}$",
        braces=["a$_{1}$–a$_{5}$", "a$_{6}$–a$_{11}$", "a$_{12}$–a$_{20}$"],
        goal="goal $z_g$",
        xlabel="planning step",
    ),
    "cn": dict(
        title_a="(a) 扁平规划",
        title_b="(b) 边界引导分割",
        err="推演误差逐步累积",
        reset="重锚定至真实观测，\n漂移清零",
        subgoal="子目标 {}",
        detect="边界检测 {}",
        brace_a="一次性联合优化 a$_1$–a$_{20}$",
        braces=["a$_1$–a$_5$", "a$_6$–a$_{11}$", "a$_{12}$–a$_{20}$"],
        goal="目标 $z_g$",
        xlabel="规划步数",
    ),
}


def draw_brace(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color=INK, lw=1.1, zorder=4)
    for xx in (x0, x1):
        ax.plot([xx, xx], [y, y + 0.10], color=INK, lw=1.1, zorder=4)


def draw_panel(ax, la, segmented):
    # error wedge (upper side only, so the area below the line stays free)
    if segmented:
        for (s0, s1) in SEGMENTS:
            ts = np.linspace(s0, s1, 120)
            e = seg_err(ts, s0, s1)
            ax.fill_between(ts, 0, e, color=WEDGE_B, lw=0, zorder=1)
            ax.plot(ts, e, color=EDGE_B, lw=0.8, alpha=0.8, zorder=2)
    else:
        ts = np.linspace(0, T_MAX, 240)
        e = err_flat(ts)
        ax.fill_between(ts, 0, e, color=WEDGE_A, lw=0, zorder=1)
        ax.plot(ts, e, color=EDGE_A, lw=0.8, alpha=0.8, zorder=2)

    # trajectory and imagined states
    ax.plot([0, T_MAX], [0, 0], color=INK, lw=1.3, zorder=3)
    dots = [x for x in range(1, 20) if not (segmented and x in (5, 11))]
    ax.scatter(dots, [0] * len(dots), s=9, color=GRAY, zorder=4)
    # start: real observation
    ax.scatter([0], [0], s=42, color=ACCENT, zorder=6, edgecolor="white", linewidth=0.8)

    # goal flag
    ax.plot([T_MAX + 0.15, T_MAX + 0.15], [0, 0.55], color=INK, lw=1.2, zorder=5)
    ax.fill([T_MAX + 0.15, T_MAX + 0.70, T_MAX + 0.15], [0.57, 0.45, 0.33],
            color=ACCENT, zorder=5)
    ax.text(T_MAX + 0.10, 0.66, la["goal"], ha="right", va="bottom",
            fontsize=9, color=INK)

    # step ticks
    for x in range(1, 20):
        ax.plot([x, x], [-0.07, -0.02], color=GRAY, lw=0.6, zorder=3)
    for x in (0, 5, 10, 15, 20):
        ax.plot([x, x], [-0.11, -0.02], color=GRAY, lw=0.8, zorder=3)
        ax.text(x, -0.15, str(x), ha="center", va="top", fontsize=8, color=GRAY)

    # optimization braces
    if segmented:
        for (s0, s1), lab in zip(SEGMENTS, la["braces"]):
            draw_brace(ax, s0, s1, -0.55)
            ax.text((s0 + s1) / 2, -0.63, lab, ha="center", va="top",
                    fontsize=8.5, color=INK)
    else:
        draw_brace(ax, 0, T_MAX, -0.55)
        ax.text(T_MAX / 2, -0.63, la["brace_a"], ha="center", va="top",
                fontsize=8.5, color=INK)

    ax.text(10.5, -1.08, la["xlabel"], ha="center", va="top",
            fontsize=8.5, color=GRAY)

    # annotations
    if segmented:
        for i, bx in enumerate(BOUNDS):
            ax.plot([bx, bx], [-0.07, 1.90], color=ACCENT, lw=1.1,
                    ls=(0, (4, 3)), zorder=2)
            ax.scatter([bx], [1.90], marker="D", s=18, color=ACCENT, zorder=6)
            ax.scatter([bx], [0], s=42, color=ACCENT, zorder=6,
                       edgecolor="white", linewidth=0.8)
            ax.text(bx, 2.02, la["detect"].format(i + 1), ha="center",
                    va="bottom", fontsize=8.5, color=ACCENT)
        for i, (s0, s1) in enumerate(SEGMENTS):
            ax.text((s0 + s1) / 2, 0.72, la["subgoal"].format(i + 1),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.annotate(la["reset"], xy=(5.0, 0.10), xytext=(6.8, 1.50),
                    fontsize=8.5, color=INK, ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    else:
        ax.annotate(la["err"], xy=(14.0, 1.0), xytext=(9.2, 2.15),
                   fontsize=8.5, color=INK, ha="left", va="top",
                   arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    ax.set_xlim(-0.6, 21.6)
    ax.set_ylim(-1.25, 2.55)
    ax.axis("off")


def build(lang):
    if lang == "en":
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        })
    else:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Songti SC", "Arial Unicode MS"],
            "axes.unicode_minus": False,
        })
    plt.rcParams["pdf.fonttype"] = 42

    la = L[lang]
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.4, 5.2))
    draw_panel(ax_a, la, segmented=False)
    ax_a.set_title(la["title_a"], loc="left", fontsize=11, fontweight="bold", pad=8)
    draw_panel(ax_b, la, segmented=True)
    ax_b.set_title(la["title_b"], loc="left", fontsize=11, fontweight="bold", pad=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02, hspace=0.55)
    stem = FIGS / f"rollout_flat_vs_seg_{lang}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"saved {stem}.pdf / .png")


if __name__ == "__main__":
    for lang in ("en", "cn"):
        build(lang)
