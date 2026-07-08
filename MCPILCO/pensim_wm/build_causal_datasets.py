"""
FILE 1 of the causal-discovery stage.

Takes the collected PenSim trajectories (a pensim_wm TrajectoryBuffer), divides
them into MDP-style datasets, and stores them in the HDF5 format the BCNP
transformer neural process consumes (CausalStructureNeuralProcess).

"Divide them in MDPs"
---------------------
Each MDP transition (s_t, a_t) -> s_{t+1} contributes one *sample* (row).
The causal-discovery "variables" (graph nodes) are, by default, the concatenation

    node vector = [ s_t (9) , a_t (6) ]   -> D = 15 nodes

so each row is one transition described over these 15 variables. We then slice
each episode into fixed-size *windows*; every window becomes one dataset
(one meta-learning example) of shape (num_samples, D). Windows are stacked into

    data  : (B, num_samples, D)      # B = number of windows
    label : (B, D, D)                # DAG adjacency per window

matching exactly what ml2_meta_causal_discovery ... MultipleFileDataset reads
(datasets "data" and "label" in each .hdf5 file).

Node-set variants (``--nodes``):
  * ``sa``       : [s_t, a_t]                           (15 nodes)  [default]
  * ``sas``      : [s_t, a_t, s_{t+1}]                  (24 nodes)
  * ``delta``    : [s_t, a_t, (s_{t+1}-s_t)]            (24 nodes)

Labels
------
BCNP is *supervised* meta-learning: training needs a known DAG per dataset.
Real PenSim trajectories have no ground-truth causal graph, so by default the
label is stored as zeros with the HDF5 attribute ``has_labels=False`` (these
datasets are inference inputs). If you have an assumed/known adjacency (domain
knowledge, or a graph learned elsewhere), pass ``--assumed_adjacency file.npy``
and it is broadcast to every window as the label.
"""

import argparse
import os

import h5py
import numpy as np

from . import config
from .buffer import TrajectoryBuffer


NODE_SETS = {
    "sa":    ("state", "action"),
    "sas":   ("state", "action", "next_state"),
    "delta": ("state", "action", "delta_state"),
}


def _node_names(node_set):
    parts = NODE_SETS[node_set]
    names = []
    for p in parts:
        if p == "state":
            names += list(config.OBS_NAMES)
        elif p == "next_state":
            names += [f"next_{n}" for n in config.OBS_NAMES]
        elif p == "delta_state":
            names += [f"d_{n}" for n in config.OBS_NAMES]
        elif p == "action":
            names += list(config.ACTION_NAMES)
    return names


def episode_to_rows(ep, node_set):
    """Build the per-transition node matrix for one episode -> (T, D)."""
    s_t = ep.obs[:-1]                    # (T, 9)
    s_tp1 = ep.obs[1:]                   # (T, 9)
    a_t = ep.actions                     # (T, 6)
    cols = []
    for p in NODE_SETS[node_set]:
        if p == "state":
            cols.append(s_t)
        elif p == "next_state":
            cols.append(s_tp1)
        elif p == "delta_state":
            cols.append(s_tp1 - s_t)
        elif p == "action":
            cols.append(a_t)
    return np.concatenate(cols, axis=1)  # (T, D)


def window_episode(rows, num_samples, stride):
    """Slice (T, D) into fixed windows (num_windows, num_samples, D)."""
    T = rows.shape[0]
    if T < num_samples:
        return np.empty((0, num_samples, rows.shape[1]), dtype=rows.dtype)
    starts = range(0, T - num_samples + 1, stride)
    return np.stack([rows[i:i + num_samples] for i in starts], axis=0)


def phase_bounds(T, breakpoint_steps):
    """Return list of (start, end, phase_id) segments within [0, T) by step index."""
    cuts = [0] + [b for b in sorted(breakpoint_steps) if 0 < b < T] + [T]
    return [(cuts[i], cuts[i + 1], i) for i in range(len(cuts) - 1)]


def phase_bounds_by_time(times, breakpoints_hours):
    """
    Return (start, end, phase_id) segments using the REAL per-transition times.
    A transition t is in phase p iff breakpoints_hours[p-1] <= times[t] < breakpoints_hours[p].
    times must be sorted ascending (it is: wall-clock within a batch).
    """
    times = np.asarray(times, dtype=np.float64)
    T = times.shape[0]
    cut_idx = [int(np.searchsorted(times, h, side="left")) for h in sorted(breakpoints_hours)]
    cuts = [0] + [c for c in cut_idx if 0 < c < T] + [T]
    # dedupe while preserving order (a breakpoint outside the time range collapses)
    seen, uniq = set(), []
    for c in cuts:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return [(uniq[i], uniq[i + 1], i) for i in range(len(uniq) - 1)]


def buffer_to_bcnp_arrays(buffer, node_set="sa", num_samples=200, stride=100,
                          breakpoint_steps=None, breakpoints_hours=None):
    """
    TrajectoryBuffer -> (data (B,N,D), phase_ids (B,), node_names).

    Phase segmentation prefers the episode's REAL times (split on
    breakpoints_hours via searchsorted) when available; otherwise falls back to
    breakpoint_steps (assumed step grid). Windows never straddle a breakpoint.
    """
    breakpoint_steps = breakpoint_steps or []
    all_windows, all_phase_ids = [], []
    used_time = False
    for ep in buffer.episodes:
        rows = episode_to_rows(ep, node_set)          # (T, D)
        T = rows.shape[0]
        if breakpoints_hours and getattr(ep, "times", None) is not None:
            segments = phase_bounds_by_time(ep.times[:T], breakpoints_hours)
            used_time = True
        else:
            segments = phase_bounds(T, breakpoint_steps)
        for (a, b, pid) in segments:
            seg = rows[a:b]
            w = window_episode(seg, num_samples, stride)
            if w.shape[0] > 0:
                all_windows.append(w)
                all_phase_ids.append(np.full(w.shape[0], pid, dtype=np.int64))
    if not all_windows:
        raise ValueError(
            f"No windows produced. Phases too short for num_samples={num_samples}? "
            f"(episode lengths: {[ep.T for ep in buffer.episodes]}, "
            f"breakpoint_steps={breakpoint_steps}, breakpoints_hours={breakpoints_hours})"
        )
    data = np.concatenate(all_windows, axis=0).astype(np.float32)  # (B, N, D)
    phase_ids = np.concatenate(all_phase_ids, axis=0)              # (B,)
    if used_time:
        print("[build] phase split used REAL per-transition times (searchsorted).")
    else:
        print("[build] phase split used assumed step grid (no times in buffer).")
    return data, phase_ids, _node_names(node_set)


def save_bcnp_hdf5(data, node_names, out_path, phase_ids=None,
                   assumed_adjacency=None, breakpoints_hours=None, name="pensim"):
    """Write BCNP-format HDF5 (datasets 'data' and 'label' + metadata attrs)."""
    B, N, D = data.shape
    if assumed_adjacency is not None:
        A = np.asarray(assumed_adjacency, dtype=np.float32)
        if A.shape != (D, D):
            raise ValueError(f"assumed_adjacency must be ({D},{D}), got {A.shape}")
        label = np.broadcast_to(A, (B, D, D)).astype(np.float32).copy()
        has_labels = True
    else:
        label = np.zeros((B, D, D), dtype=np.float32)
        has_labels = False

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data)       # (B, N, D)
        f.create_dataset("label", data=label)     # (B, D, D)
        if phase_ids is not None:
            f.create_dataset("phase_ids", data=np.asarray(phase_ids, dtype=np.int64))
        f.attrs["has_labels"] = has_labels
        f.attrs["num_nodes"] = D
        f.attrs["num_samples"] = N
        f.attrs["num_datasets"] = B
        f.attrs["node_names"] = np.array(node_names, dtype=h5py.string_dtype())
        if breakpoints_hours is not None:
            f.attrs["breakpoints_hours"] = np.asarray(breakpoints_hours, dtype=np.float64)
    return dict(B=B, N=N, D=D, has_labels=has_labels)


def main():
    p = argparse.ArgumentParser(description="Buffer -> BCNP-format causal datasets (HDF5)")
    p.add_argument("--buffer", type=str, required=True)
    p.add_argument("--out", type=str, default="datasets/data/pensim_causal/pensim.hdf5")
    p.add_argument("--nodes", type=str, default="sa", choices=list(NODE_SETS.keys()))
    p.add_argument("--num_samples", type=int, default=200,
                   help="rows (transitions) per dataset window")
    p.add_argument("--stride", type=int, default=100,
                   help="window stride (use < num_samples for overlap / more windows)")
    p.add_argument("--breakpoints_hours", type=float, nargs="*", default=None,
                   help="phase breakpoints in HOURS (default: config PHASE_BREAKPOINTS_HOURS). "
                        "Pass empty to disable phase-aware windowing.")
    p.add_argument("--step_size_hours", type=float, default=config.STEP_SIZE_HOURS)
    p.add_argument("--split_per_phase", action="store_true",
                   help="also write one HDF5 per phase next to --out")
    p.add_argument("--assumed_adjacency", type=str, default=None,
                   help="optional .npy (D,D) DAG used as label for every window")
    args = p.parse_args()

    buf = TrajectoryBuffer.load(args.buffer)
    print(f"[build] buffer: {len(buf)} episodes, {buf.total_transitions} transitions")

    bph = config.PHASE_BREAKPOINTS_HOURS if args.breakpoints_hours is None else args.breakpoints_hours
    bp_steps = config.breakpoints_to_steps(bph, args.step_size_hours)
    print(f"[build] phase breakpoints: {bph} h -> steps {bp_steps} "
          f"(step size {args.step_size_hours} h)")

    data, phase_ids, node_names = buffer_to_bcnp_arrays(
        buf, node_set=args.nodes, num_samples=args.num_samples,
        stride=args.stride, breakpoint_steps=bp_steps, breakpoints_hours=bph,
    )
    A = np.load(args.assumed_adjacency) if args.assumed_adjacency else None
    info = save_bcnp_hdf5(data, node_names, args.out, phase_ids=phase_ids,
                          assumed_adjacency=A, breakpoints_hours=bph)

    print(f"[build] node set '{args.nodes}': {info['D']} nodes -> {node_names}")
    print(f"[build] wrote {info['B']} datasets x {info['N']} samples x {info['D']} nodes")
    uniq, counts = np.unique(phase_ids, return_counts=True)
    print(f"[build] windows per phase: " +
          ", ".join(f"phase{u}={c}" for u, c in zip(uniq, counts)))
    print(f"[build] has_labels={info['has_labels']}  -> {args.out}")

    if args.split_per_phase:
        base, ext = os.path.splitext(args.out)
        for u in uniq:
            sel = phase_ids == u
            pinfo = save_bcnp_hdf5(
                data[sel], node_names, f"{base}_phase{u}{ext}",
                phase_ids=phase_ids[sel], assumed_adjacency=A, breakpoints_hours=bph)
            print(f"[build]   phase {u}: {pinfo['B']} datasets -> {base}_phase{u}{ext}")

    if not info["has_labels"]:
        print("[build] note: no DAG labels stored (inference inputs). Pass "
              "--assumed_adjacency to attach a known/assumed graph for supervised training.")


if __name__ == "__main__":
    main()
