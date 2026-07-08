"""
Collect the initial *offline* buffer of sub-optimal PenSim trajectories.

The environment is injected via a ``make_env`` callable so this module never
imports smpl/pensimpy directly -- that keeps it unit-testable with a mock env

Each rollout perturbs the baseline recipe action within +/-10% of the setpoint
range, producing physically-valid but clearly sub-optimal batches.
"""

import argparse
from typing import Callable, Optional

import numpy as np

from . import config
from .buffer import TrajectoryBuffer, perturbed_action


def rollout_episode(env, rng, rel_scale: float = 0.1, max_steps: Optional[int] = None):
    """Run one perturbed-baseline episode; return (obs, actions, rewards, dones)."""
    max_steps = config.STEPS_PER_EPISODE if max_steps is None else max_steps

    obs0 = env.reset()
    obs_list = [np.asarray(obs0, dtype=np.float64).reshape(-1)]
    act_list, rew_list, done_list = [], [], []

    for _ in range(max_steps):
        a = perturbed_action(rng, rel_scale=rel_scale)
        obs, reward, done, _info = env.step(a)
        obs_list.append(np.asarray(obs, dtype=np.float64).reshape(-1))
        act_list.append(a)
        rew_list.append(float(reward))
        done_list.append(bool(done))
        if done:
            break

    return (
        np.stack(obs_list, 0),
        np.stack(act_list, 0),
        np.asarray(rew_list, dtype=np.float64),
        np.asarray(done_list, dtype=bool),
    )


def collect_offline_buffer(
    make_env: Callable[[], object],
    n_episodes: int = 5,
    seed: int = 0,
    rel_scale: float = 0.1,
    max_steps: Optional[int] = None,
    out_path: Optional[str] = None,
    verbose: bool = True,
) -> TrajectoryBuffer:
    """Collect ``n_episodes`` sub-optimal rollouts into a TrajectoryBuffer."""
    rng = np.random.default_rng(seed)
    env = make_env()
    buf = TrajectoryBuffer()

    for ep in range(n_episodes):
        obs, acts, rews, dones = rollout_episode(env, rng, rel_scale, max_steps)
        buf.add_episode(obs, acts, rews, dones)
        if verbose:
            print(
                f"[collect] episode {ep + 1}/{n_episodes}: "
                f"T={acts.shape[0]}  total_yield={rews.sum():.2f}  "
                f"final_yield_step={rews[-1]:.4f}"
            )

    if verbose:
        print(f"[collect] buffer: {len(buf)} episodes, "
              f"{buf.total_transitions} transitions")
    if out_path is not None:
        buf.save(out_path)
        if verbose:
            print(f"[collect] saved -> {out_path}")
    return buf


def _build_default_env():
    """Build the baseline PenSim env exactly like demo.py (imported lazily)."""
    from smpl.envs.pensimenv import PenSimEnvGym
    from pensimpy.examples.recipe import Recipe, RecipeCombo
    from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
    from pensimpy.data.constants import (
        FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
        PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE,
        PAA_DEFAULT_PROFILE,
    )

    recipe_dict = {
        FS: Recipe(FS_DEFAULT_PROFILE, FS),
        FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
        FG: Recipe(FG_DEFAULT_PROFILE, FG),
        PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
        DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
        WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
        PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
    }
    recipe_combo = RecipeCombo(recipe_dict=recipe_dict)

    def raw_yield_reward(previous_observation, action, current_observation, reward=None):
        return reward  # yield-per-step already computed by pensimpy

    return PenSimEnvGym(
        recipe_combo=recipe_combo,
        reward_function=raw_yield_reward,
        dense_reward=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect sub-optimal PenSim offline buffer")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rel_scale", type=float, default=0.1, help="perturbation as fraction of setpoint range")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--out", type=str, default="buffers/pensim_offline.pkl")
    args = p.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    collect_offline_buffer(
        _build_default_env,
        n_episodes=args.n_episodes,
        seed=args.seed,
        rel_scale=args.rel_scale,
        max_steps=args.max_steps,
        out_path=args.out,
    )
