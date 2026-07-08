"""
Ingest smpl's pre-generated (sub-optimal) PenSim batches into a TrajectoryBuffer.

smpl ships example batches at ``smpl/.../examples/example_batches`` and exposes
``PeniControlData`` which returns a d4rl-style flat dict. These batches are
produced from the baseline recipe (no gpei optimization), so they ARE the
"sub-optimal trajectories from smpl" we want as the initial offline dataset.

This module is defensive about the exact dict keys (d4rl variants differ), and
prints what it finds so we can confirm the structure on the real cluster. Load
with normalize=False so the buffer stores raw physical units -- the world model
standardizes internally later.
"""

import argparse
import os

import numpy as np

from . import config
from .buffer import TrajectoryBuffer


# ---------------------------------------------------------------------------
# locate the example-batches folder that ships with smpl
# ---------------------------------------------------------------------------
def find_example_batches(explicit=None):
    if explicit and os.path.isdir(explicit):
        return explicit
    candidates = []
    if explicit:
        candidates.append(explicit)
    try:
        import smpl
        smpl_root = os.path.dirname(os.path.abspath(smpl.__file__))
        candidates += [
            os.path.join(smpl_root, "configdata", "pensimenv"),   # actual smpl layout
            os.path.join(smpl_root, "examples", "example_batches"),
            os.path.join(smpl_root, "..", "examples", "example_batches"),
            os.path.join(smpl_root, "envs", "examples", "example_batches"),
        ]
    except Exception:
        pass
    candidates += [
        "examples/example_batches",
        os.path.join(os.getcwd(), "smpl", "smpl", "configdata", "pensimenv"),
        os.path.join(os.getcwd(), "smpl", "examples", "example_batches"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)

    # last resort: search for any directory (excluding build/) containing
    # csv files with "batch" in the name.
    try:
        import smpl
        search_root = os.path.dirname(os.path.dirname(os.path.abspath(smpl.__file__)))
        best = None
        for dirpath, dirnames, filenames in os.walk(search_root):
            if "build" in dirpath.split(os.sep):
                continue
            csvs = [f for f in filenames if f.endswith(".csv") and "batch" in f.lower()]
            if csvs:
                best = dirpath
                break
        if best:
            return best
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not locate smpl example_batches. Pass --batches_dir explicitly. "
        f"Tried: {[os.path.abspath(c) for c in candidates if c]}"
    )


# ---------------------------------------------------------------------------
# robust key lookup over d4rl-style dicts
# ---------------------------------------------------------------------------
def _pick(d, names):
    for n in names:
        if n in d:
            return np.asarray(d[n])
    return None


def read_time_grid(batches_dir, verbose=True):
    """
    Read the actual 'Time Step' column (hours) from a batch CSV.

    All PenSim batch CSVs share the identical time grid (0.2 .. 230.0 h in 0.2 h
    steps, with a header row and data starting at 0.2 h), so reading one file's
    time column gives the wall-clock time of every transition. Returns a 1-D
    array of length = number of data rows, or None if no CSV is found/parseable.
    """
    import glob
    csvs = sorted(glob.glob(os.path.join(batches_dir, "*.csv")))
    if not csvs:
        if verbose:
            print(f"[ingest] no CSV in {batches_dir}; cannot read real time grid.")
        return None
    try:
        # column 0 = "Time Step"; skip the header row.
        times = np.genfromtxt(csvs[0], delimiter=",", skip_header=1, usecols=0)
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if verbose:
            print(f"[ingest] time grid from {os.path.basename(csvs[0])}: "
                  f"{times.shape[0]} rows, t=[{times[0]:.3g} .. {times[-1]:.3g}] h, "
                  f"step~{np.median(np.diff(times)):.3g} h")
        return times
    except Exception as e:
        if verbose:
            print(f"[ingest] could not parse time column from {csvs[0]}: {e}")
        return None


def dataset_dict_to_buffer(ds, episode_len=None, verbose=True, time_grid=None):
    """Convert a d4rl-style flat dict into a TrajectoryBuffer (episode-segmented)."""
    obs = _pick(ds, ["observations", "obs", "states"])
    act = _pick(ds, ["actions", "acts"])
    rew = _pick(ds, ["rewards", "reward", "yield_per_step"])
    term = _pick(ds, ["terminals", "dones", "done", "terminal"])
    next_obs = _pick(ds, ["next_observations", "next_obs"])

    if obs is None or act is None or rew is None:
        raise KeyError(
            f"Dataset missing required keys. Present keys: {list(ds.keys())}"
        )

    obs = obs.astype(np.float64)
    act = act.astype(np.float64)
    rew = rew.astype(np.float64).reshape(-1)
    N = obs.shape[0]

    if verbose:
        print(f"[ingest] flat dataset: {N} transitions, "
              f"obs{obs.shape}, act{act.shape}, rew{rew.shape}, "
              f"terminals={'yes' if term is not None else 'no'}, "
              f"next_obs={'yes' if next_obs is not None else 'no'}")

    # --- episode boundaries -------------------------------------------------
    if term is not None and np.asarray(term).reshape(-1).sum() > 0:
        term = np.asarray(term).reshape(-1).astype(bool)
        ends = np.where(term)[0] + 1                     # exclusive ends
        if ends[-1] != N:
            ends = np.append(ends, N)
        bounds = np.concatenate([[0], ends])
    else:
        L = episode_len or config.STEPS_PER_EPISODE
        if N % L != 0 and verbose:
            print(f"[ingest] warning: {N} not divisible by episode_len {L}; "
                  f"last episode will be shorter.")
        bounds = np.arange(0, N + 1, L)
        if bounds[-1] != N:
            bounds = np.append(bounds, N)

    # --- build episodes -----------------------------------------------------
    buf = TrajectoryBuffer()
    for i0, i1 in zip(bounds[:-1], bounds[1:]):
        if i1 <= i0:
            continue
        o = obs[i0:i1]                                   # (L, OBS)
        a = act[i0:i1]                                   # (L, ACT)
        r = rew[i0:i1]                                   # (L,)
        d = (term[i0:i1] if term is not None
             else np.zeros(i1 - i0, dtype=bool))
        if next_obs is not None:
            full_obs = np.concatenate([o, next_obs[i1 - 1][None, :]], 0)  # (L+1, OBS)
        else:
            # last next-state unknown; duplicate last obs (flagged)
            full_obs = np.concatenate([o, o[-1][None, :]], 0)
        # real wall-clock time of each transition's s_t: all batches share the
        # same grid, so take the first L entries (episodes are 0-based within-batch).
        L = i1 - i0
        if time_grid is not None and len(time_grid) >= L:
            times = np.asarray(time_grid[:L], dtype=np.float64)
        else:
            times = None
        buf.add_episode(full_obs, a, r, d, times=times)

    if verbose:
        print(f"[ingest] -> {len(buf)} episodes, {buf.total_transitions} transitions")
        if next_obs is None:
            print("[ingest] note: no next_observations in source; each episode's "
                  "final transition uses a duplicated last state (delta ~ 0 there).")
        if time_grid is not None:
            print("[ingest] attached REAL per-transition times from CSV time column.")
        else:
            print("[ingest] warning: no time grid attached; phase splits will fall "
                  "back to assumed step size.")
    return buf


def load_smpl_dataset(batches_dir=None, normalize=False, episode_len=None, verbose=True):
    """Load smpl example batches via PeniControlData into a TrajectoryBuffer."""
    from smpl.envs.pensimenv import PeniControlData

    folder = find_example_batches(batches_dir)
    if verbose:
        print(f"[ingest] loading smpl batches from: {folder}")
    data = PeniControlData(
        dataset_folder=folder,
        observation_dim=config.OBS_DIM,
        action_dim=config.ACTION_DIM,
        normalize=normalize,
    )
    ds = data.get_dataset()
    if not isinstance(ds, dict):
        raise TypeError(f"get_dataset() returned {type(ds)}, expected dict")
    if verbose:
        print("[ingest] get_dataset() keys and shapes:")
        for k, v in ds.items():
            try:
                print(f"    {k}: shape={np.asarray(v).shape} dtype={np.asarray(v).dtype}")
            except Exception:
                print(f"    {k}: (unprintable) type={type(v)}")
    time_grid = read_time_grid(folder, verbose=verbose)
    return dataset_dict_to_buffer(ds, episode_len=episode_len, verbose=verbose,
                                  time_grid=time_grid)


def main():
    p = argparse.ArgumentParser(description="Ingest smpl sub-optimal batches -> TrajectoryBuffer")
    p.add_argument("--batches_dir", type=str, default=None,
                   help="path to smpl example_batches (auto-located if omitted)")
    p.add_argument("--out", type=str, default="buffers/smpl_offline.pkl")
    p.add_argument("--normalize", action="store_true",
                   help="store env-normalized obs instead of raw physical units")
    p.add_argument("--episode_len", type=int, default=None,
                   help="fallback episode length if no terminals in dataset")
    args = p.parse_args()

    buf = load_smpl_dataset(
        batches_dir=args.batches_dir,
        normalize=args.normalize,
        episode_len=args.episode_len,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    buf.save(args.out)
    print(f"[ingest] saved buffer -> {args.out}")


if __name__ == "__main__":
    main()
