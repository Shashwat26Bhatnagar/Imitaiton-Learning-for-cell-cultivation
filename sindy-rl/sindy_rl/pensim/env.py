"""PenSim (smpl) for SINDy-RL: gymnasium API, ZOH frame-skip, residual-on-recipe actions."""
import numpy as np
import gymnasium
from gymnasium.spaces import Box

from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE, PRESS_DEFAULT_PROFILE,
    DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE)
from smpl.envs.pensimenv import PenSimEnvGym, NUM_STEPS, STEP_IN_MINUTES
from smpl.envs.utils import normalize_spaces

OBS_NAMES = ['t', 'pH', 'T', 'Fa', 'Fb', 'Fc', 'Fh', 'Wt', 'DO2']
ACT_NAMES = ['discharge', 'Fs', 'Foil', 'Fg', 'pres', 'Fw']

# smpl's action order vs pensimpy's recipe keys
_RECIPE_KEYS = [DISCHARGE, FS, FOIL, FG, PRES, WATER]


def default_recipe_combo():
    return RecipeCombo(recipe_dict={
        FS: Recipe(FS_DEFAULT_PROFILE, FS),
        FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
        FG: Recipe(FG_DEFAULT_PROFILE, FG),
        PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
        DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
        WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
        PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
    })


class PenSimSINDyEnv(gymnasium.Env):
    """
    act: residual on the industrial recipe.
        a_t = clip(recipe(t) + alpha * u_t * (hi - lo), lo, hi),  u_t in [-1,1]^6
    Absolute actions are infeasible for random exploration -- the solver dies and
    smpl returns error_reward=-100. The residual keeps rollouts on the feasible
    manifold; u=0 reproduces the ~3224 baseline batch.

    obs: [t, pH, T, Fa, Fb, Fc, Fh, Wt, DO2] x (1 + n_delay), newest block first.
    rew: penicillin yield per step (pensimpy's yield_per_run).
    """

    def __init__(self, env_config=None):
        cfg = dict(env_config or {})
        self.frame_skip = int(cfg.get('frame_skip', 5))     # 12 min -> 1 h
        self.n_delay = int(cfg.get('n_delay', 0))
        self.alpha = float(cfg.get('residual_alpha', 0.1))
        self.seed_ref = cfg.get('random_seed_ref', None)

        self.recipe = default_recipe_combo()
        self.env = PenSimEnvGym(recipe_combo=self.recipe, normalize=True, fast=True)

        self.lo = np.asarray(self.env.min_actions, dtype=np.float32)
        self.hi = np.asarray(self.env.max_actions, dtype=np.float32)
        self.span = self.hi - self.lo

        self.max_episode_steps = int(
            cfg.get('max_episode_steps', NUM_STEPS // self.frame_skip))

        self.observation_space = Box(-np.inf, np.inf,
                                     shape=(9 * (1 + self.n_delay),), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        self._hist = []
        self.n_episode_steps = 0

    def _recipe_action(self):
        """Baseline setpoints at the current time, in smpl's action order.
        Uses STEP_IN_MINUTES to match smpl's own internal recipe lookup exactly."""
        t = self.env.step_count * STEP_IN_MINUTES
        vals = self.recipe.get_values_dict_at(t)
        return np.array([vals[k] for k in _RECIPE_KEYS], dtype=np.float32)

    def _stack(self, obs):
        self._hist.append(np.asarray(obs, dtype=np.float32))
        while len(self._hist) < self.n_delay + 1:
            self._hist.insert(0, self._hist[0])
        self._hist = self._hist[-(self.n_delay + 1):]
        return np.concatenate(self._hist[::-1])

    def reset(self, seed=None, options=None):
        self._hist = []
        self.n_episode_steps = 0
        obs = self.env.reset(random_seed_ref=self.seed_ref)
        return self._stack(obs), {}

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        rew = 0.0
        done = False
        info = {}
        obs = None
        for _ in range(self.frame_skip):
            a_abs = np.clip(self._recipe_action() + self.alpha * u * self.span,
                            self.lo, self.hi)
            a_norm, _, _ = normalize_spaces(a_abs, self.hi, self.lo)
            obs, r, done, info = self.env.step(a_norm.tolist())   # old gym 4-tuple
            rew += float(r)                                       # yield per step
            if done:
                break

        self.n_episode_steps += 1
        term = bool(info.get('error_occurred', False))
        trunc = bool(done) or (self.n_episode_steps >= self.max_episode_steps)
        return self._stack(obs), rew, term, trunc, info
