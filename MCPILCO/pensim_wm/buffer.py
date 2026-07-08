"""
TrajectoryBuffer: the initial *offline* dataset of (sub-optimal) PenSim rollouts.

A "trajectory" / "batch" / "episode" is stored as four aligned arrays following
the standard RL convention  (s_t, a_t) -> s_{t+1}, r_t :

    obs      : (T+1, OBS_DIM)   observations o_0 .. o_T  (includes reset obs)
    actions  : (T,   ACTION_DIM) raw actions a_0 .. a_{T-1}
    rewards  : (T,)             yield-per-step r_0 .. r_{T-1}
    dones    : (T,)             episode-termination flags

The buffer is deliberately dumb (just storage + dataset assembly). Anything that
needs the environment lives in collect_data.py; anything GP-specific lives in
world_model.py. This keeps it reusable for the later BCNP / policy stages.
"""

import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import config


@dataclass
class Episode:
    obs: np.ndarray       # (T+1, OBS_DIM)
    actions: np.ndarray   # (T, ACTION_DIM)
    rewards: np.ndarray   # (T,)
    dones: np.ndarray     # (T,)
    times: Optional[np.ndarray] = None   # (T,) wall-clock time (hours) of each transition's s_t

    def __post_init__(self):
        self.obs = np.asarray(self.obs, dtype=np.float64)
        self.actions = np.asarray(self.actions, dtype=np.float64)
        self.rewards = np.asarray(self.rewards, dtype=np.float64).reshape(-1)
        self.dones = np.asarray(self.dones).reshape(-1)
        T = self.actions.shape[0]
        assert self.obs.shape[0] == T + 1, f"obs must be T+1 rows, got {self.obs.shape[0]} vs T={T}"
        assert self.rewards.shape[0] == T
        assert self.dones.shape[0] == T
        if self.times is not None:
            self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
            assert self.times.shape[0] == T, \
                f"times must have T={T} entries, got {self.times.shape[0]}"

    @property
    def T(self) -> int:
        return self.actions.shape[0]


@dataclass
class TrajectoryBuffer:
    episodes: List[Episode] = field(default_factory=list)

    # -- construction ---------------------------------------------------------
    def add_episode(self, obs, actions, rewards, dones, times=None) -> None:
        self.episodes.append(Episode(obs, actions, rewards, dones, times))

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def total_transitions(self) -> int:
        return sum(ep.T for ep in self.episodes)

    # -- persistence ----------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "episodes": [
                        (ep.obs, ep.actions, ep.rewards, ep.dones, ep.times)
                        for ep in self.episodes
                    ]
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "TrajectoryBuffer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        buf = cls()
        for rec in d["episodes"]:
            if len(rec) == 5:                       # new format with times
                obs, actions, rewards, dones, times = rec
            else:                                    # old 4-tuple buffers
                obs, actions, rewards, dones = rec
                times = None
            buf.add_episode(obs, actions, rewards, dones, times)
        return buf

    # -- dataset assembly -----------------------------------------------------
    def stacked_transitions(self, stride: int = 1, max_points: Optional[int] = None):
        """
        Flatten every episode into pointwise transition tensors:

            X_obs  : (N, OBS_DIM)     o_t
            X_act  : (N, ACTION_DIM)  a_t
            Y_next : (N, OBS_DIM)     o_{t+1}
            Y_rew  : (N,)             r_t

        ``stride`` subsamples within each episode (GP cost is O(N^3), and PenSim
        episodes are 1150 steps, so subsampling is usually needed).
        ``max_points`` caps the total after a final uniform subsample.
        """
        Xo, Xa, Yn, Yr = [], [], [], []
        for ep in self.episodes:
            idx = np.arange(0, ep.T, stride)
            Xo.append(ep.obs[idx])            # o_t
            Xa.append(ep.actions[idx])        # a_t
            Yn.append(ep.obs[idx + 1])        # o_{t+1}
            Yr.append(ep.rewards[idx])        # r_t
        Xo = np.concatenate(Xo, 0)
        Xa = np.concatenate(Xa, 0)
        Yn = np.concatenate(Yn, 0)
        Yr = np.concatenate(Yr, 0)

        if max_points is not None and Xo.shape[0] > max_points:
            sel = np.linspace(0, Xo.shape[0] - 1, max_points).round().astype(int)
            Xo, Xa, Yn, Yr = Xo[sel], Xa[sel], Yn[sel], Yr[sel]

        return Xo, Xa, Yn, Yr

    def episode_arrays(self, stride: int = 1):
        """
        Yield per-episode (obs, actions) arrays (obs length T'+1, actions T'+1
        padded) suitable for MC-PILCO's ``Model_learning.add_data`` which computes
        deltas *within* each call and so must not span episode boundaries.

        The last action row is a duplicate padding that ``data_to_gp_IO`` drops.
        """
        for ep in self.episodes:
            idx = np.arange(0, ep.T, stride)
            obs = np.concatenate([ep.obs[idx], ep.obs[idx[-1] + 1][None, :]], 0)  # (n+1, OBS)
            acts = ep.actions[idx]                                                # (n, ACT)
            acts_padded = np.concatenate([acts, acts[-1][None, :]], 0)            # (n+1, ACT)
            yield obs, acts_padded


def perturbed_action(rng, nominal=None, rel_scale=0.1) -> np.ndarray:
    """
    Draw a sub-optimal action: baseline setpoint + uniform noise within
    +/- ``rel_scale`` of the setpoint range, clipped to valid bounds.

    ``rel_scale=0.1`` matches the +/-10% input search space the PenSim docs
    describe around setpoint inputs.
    """
    nominal = config.NOMINAL_ACTION if nominal is None else np.asarray(nominal, dtype=np.float64)
    span = (config.MAX_ACTION - config.MIN_ACTION)
    noise = (rng.random(config.ACTION_DIM) * 2.0 - 1.0) * rel_scale * span
    return config.clip_action(nominal + noise)
