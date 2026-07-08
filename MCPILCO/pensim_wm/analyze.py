"""
Analyze a PenSim TrajectoryBuffer of (sub-optimal) trajectories.

Produces:
  * report.md               -- dataset summary, per-channel & per-action stats,
                               yield summary vs the ~3640 kg baseline
  * yield_per_step.png      -- reward over time, one line per episode + mean
  * cumulative_yield.png    -- total yield per episode (bar) vs baseline ref
  * obs_trajectories.png    -- each of the 9 obs channels over time (mean +/- band)
  * obs_correlation.png     -- 9x9 observation correlation heatmap
  * action_ranges.png       -- action usage per dim vs valid bounds + nominal
  * yield_hist.png          -- distribution of yield-per-step

Headless-safe (Agg backend), so it runs on a login/compute node with no display.
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config             # noqa: E402
from .buffer import TrajectoryBuffer  # noqa: E402

# Documented baseline batch yield (kg) from the PenSim docs, for reference.
BASELINE_BATCH_YIELD_KG = 3640.0


def _basic_stats(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(min=float(x.min()), max=float(x.max()),
                mean=float(x.mean()), std=float(x.std()),
                median=float(np.median(x)))


def compute_stats(buffer: TrajectoryBuffer):
    Xo, Xa, _, Yr = buffer.stacked_transitions(stride=1)
    per_obs = [_basic_stats(Xo[:, i]) for i in range(Xo.shape[1])]
    per_act = [_basic_stats(Xa[:, i]) for i in range(Xa.shape[1])]
    ep_yields = np.array([ep.rewards.sum() for ep in buffer.episodes])
    return {
        "n_episodes": len(buffer),
        "n_transitions": buffer.total_transitions,
        "obs_dim": Xo.shape[1],
        "act_dim": Xa.shape[1],
        "per_obs": per_obs,
        "per_act": per_act,
        "reward": _basic_stats(Yr),
        "episode_yields": ep_yields,
        "Xo": Xo, "Xa": Xa, "Yr": Yr,
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def _plot_yield_per_step(buffer, path):
    plt.figure(figsize=(8, 4.5))
    maxT = max(ep.T for ep in buffer.episodes)
    acc = np.full((len(buffer), maxT), np.nan)
    for k, ep in enumerate(buffer.episodes):
        plt.plot(np.arange(ep.T), ep.rewards, color="steelblue", alpha=0.25, lw=0.8)
        acc[k, :ep.T] = ep.rewards
    plt.plot(np.arange(maxT), np.nanmean(acc, 0), color="crimson", lw=2, label="mean")
    plt.xlabel("step"); plt.ylabel("yield per step")
    plt.title("Yield-per-step over time (per episode)")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_cumulative_yield(stats, path):
    ey = stats["episode_yields"]
    plt.figure(figsize=(8, 4.5))
    plt.bar(np.arange(len(ey)), ey, color="seagreen", alpha=0.8)
    plt.axhline(BASELINE_BATCH_YIELD_KG, color="crimson", ls="--",
                label=f"documented baseline ~{BASELINE_BATCH_YIELD_KG:.0f}")
    plt.xlabel("episode"); plt.ylabel("total yield (sum of yield-per-step)")
    plt.title("Total yield per episode"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_obs_trajectories(buffer, path):
    D = config.OBS_DIM
    maxT = max(ep.T + 1 for ep in buffer.episodes)
    acc = np.full((len(buffer), maxT, D), np.nan)
    for k, ep in enumerate(buffer.episodes):
        acc[k, :ep.obs.shape[0], :] = ep.obs
    ncol = 3; nrow = int(np.ceil(D / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    t = np.arange(maxT)
    for i in range(D):
        m = np.nanmean(acc[:, :, i], 0)
        s = np.nanstd(acc[:, :, i], 0)
        axes[i].plot(t, m, color="navy", lw=1.2)
        axes[i].fill_between(t, m - s, m + s, color="navy", alpha=0.2)
        axes[i].set_title(config.OBS_NAMES[i], fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(D, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Observation channels over time (mean +/- std across episodes)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_obs_correlation(stats, path):
    Xo = stats["Xo"]
    C = np.corrcoef(Xo.T)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(config.OBS_DIM), config.OBS_NAMES, rotation=90, fontsize=7)
    plt.yticks(range(config.OBS_DIM), config.OBS_NAMES, fontsize=7)
    plt.title("Observation correlation")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_action_ranges(stats, path):
    Xa = stats["Xa"]
    A = config.ACTION_DIM
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # normalize each action dim to its [min,max] bound for comparable display
    lo, hi = config.MIN_ACTION, config.MAX_ACTION
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    norm = (Xa - lo) / span
    ax.boxplot([norm[:, i] for i in range(A)], positions=range(A), widths=0.6)
    nom = (config.NOMINAL_ACTION - lo) / span
    ax.plot(range(A), nom, "r*", markersize=12, label="nominal baseline")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(A))
    ax.set_xticklabels(config.ACTION_NAMES, rotation=45, fontsize=8)
    ax.set_ylabel("fraction of [min,max] range")
    ax.set_title("Action usage vs valid bounds"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_yield_hist(stats, path):
    plt.figure(figsize=(7, 4.5))
    plt.hist(stats["Yr"], bins=60, color="darkorange", alpha=0.85)
    plt.xlabel("yield per step"); plt.ylabel("count")
    plt.title("Distribution of yield-per-step")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt_row(name, s):
    return (f"| {name} | {s['min']:.4g} | {s['max']:.4g} | "
            f"{s['mean']:.4g} | {s['std']:.4g} | {s['median']:.4g} |")


def write_report(stats, out_dir, path):
    ey = stats["episode_yields"]
    lines = []
    lines.append("# PenSim sub-optimal trajectory analysis\n")
    lines.append(f"- Episodes: **{stats['n_episodes']}**")
    lines.append(f"- Transitions: **{stats['n_transitions']}**")
    lines.append(f"- Observation dim: {stats['obs_dim']}, Action dim: {stats['act_dim']}\n")

    lines.append("## Yield summary\n")
    lines.append(f"- Yield-per-step: mean {stats['reward']['mean']:.4g}, "
                 f"std {stats['reward']['std']:.4g}, "
                 f"range [{stats['reward']['min']:.4g}, {stats['reward']['max']:.4g}]")
    lines.append(f"- Total yield per episode: mean {ey.mean():.4g}, "
                 f"std {ey.std():.4g}, range [{ey.min():.4g}, {ey.max():.4g}]")
    lines.append(f"- Documented baseline batch yield: ~{BASELINE_BATCH_YIELD_KG:.0f} kg "
                 "(reference for suboptimality)\n")

    lines.append("## Per-observation-channel statistics (raw units)\n")
    lines.append("| channel | min | max | mean | std | median |")
    lines.append("|---|---|---|---|---|---|")
    for i, s in enumerate(stats["per_obs"]):
        lines.append(_fmt_row(config.OBS_NAMES[i], s))
    lines.append("")
    scales = [s["max"] - s["min"] for s in stats["per_obs"]]
    scales = [x for x in scales if x > 0]
    if scales:
        lines.append(f"> Channel ranges span ~{min(scales):.3g} to ~{max(scales):.3g} "
                     f"(ratio ~{max(scales)/max(min(scales),1e-9):.3g}x) -- motivates "
                     "per-dimension standardization before GP fitting.\n")

    lines.append("## Per-action statistics (raw units)\n")
    lines.append("| action | min | max | mean | std | median |")
    lines.append("|---|---|---|---|---|---|")
    for i, s in enumerate(stats["per_act"]):
        lines.append(_fmt_row(config.ACTION_NAMES[i], s))
    lines.append("")

    lines.append("## Figures\n")
    for fn in ["yield_per_step.png", "cumulative_yield.png", "obs_trajectories.png",
               "obs_correlation.png", "action_ranges.png", "yield_hist.png"]:
        lines.append(f"- `{fn}`")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def analyze_buffer(buffer: TrajectoryBuffer, out_dir: str, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    stats = compute_stats(buffer)
    _plot_yield_per_step(buffer, os.path.join(out_dir, "yield_per_step.png"))
    _plot_cumulative_yield(stats, os.path.join(out_dir, "cumulative_yield.png"))
    _plot_obs_trajectories(buffer, os.path.join(out_dir, "obs_trajectories.png"))
    _plot_obs_correlation(stats, os.path.join(out_dir, "obs_correlation.png"))
    _plot_action_ranges(stats, os.path.join(out_dir, "action_ranges.png"))
    _plot_yield_hist(stats, os.path.join(out_dir, "yield_hist.png"))
    report_path = os.path.join(out_dir, "report.md")
    write_report(stats, out_dir, report_path)
    if verbose:
        print(f"[analyze] wrote report + 6 figures to {out_dir}/")
        print(f"[analyze] episodes={stats['n_episodes']}, "
              f"transitions={stats['n_transitions']}, "
              f"mean episode yield={stats['episode_yields'].mean():.4g}")
    return stats


def main():
    p = argparse.ArgumentParser(description="Analyze a PenSim TrajectoryBuffer")
    p.add_argument("--buffer", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="analysis_out")
    args = p.parse_args()
    buf = TrajectoryBuffer.load(args.buffer)
    analyze_buffer(buf, args.out_dir)


if __name__ == "__main__":
    main()
