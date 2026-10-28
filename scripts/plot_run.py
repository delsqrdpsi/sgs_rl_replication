"""Plot a run's training curves from its TensorBoard events: python scripts/plot_run.py runs/<name> out.png"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FAMILY_COLORS = {"near_goal": "#2a78d6", "stable_grasp": "#eb6834", "reaching": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def smooth(y, k=9):
    if len(y) < k:
        return np.asarray(y)
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def main(run_dir, out):
    events = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    events.Reload()
    series = lambda tag: tuple(map(np.array, zip(*[(s.step, s.value) for s in events.Scalars(tag)])))

    panels = [
        ("Success rate by start type (training episodes)", [(f"Episode/success_{f}", f) for f in FAMILY_COLORS], (0, 1)),
        ("Share of resets SGS gives each start type", [(f"Sampler/share_{f}", f) for f in FAMILY_COLORS], (0, 1)),
        ("Held-out Reaching success (deterministic policy)", [("Eval/success_reaching", "reaching")], (0, None)),
        ("Failure rate (rod dropped out of workspace or arm too fast)", [("Episode/failure", None)], (0, None)),
        ("Mean episode length (policy steps, max 160)", [("Train/mean_episode_length", None)], (0, None)),
        ("Exploration noise (policy std)", [("Policy/mean_std", None)], (0, None)),
    ]
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    for ax, (title, lines, ylim) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        for tag, family in lines:
            steps, values = series(tag)
            color = FAMILY_COLORS.get(family, "#2a78d6")
            marker = "o" if len(steps) < 20 else None
            ax.plot(steps, smooth(values) if marker is None else values, color=color, lw=2, marker=marker,
                    markersize=8, label=family.replace("_", " ") if len(lines) > 1 else None)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("training iteration")
        ax.set_ylim(*ylim)
        ax.grid(color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        if len(lines) > 1:
            ax.legend(frameon=False)
    fig.suptitle(run_dir.rstrip("/").split("/")[-1], x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)


if __name__ == "__main__":
    main(*sys.argv[1:3])
