'''
sindy_rl/pensim/normalize.py

NON-DESTRUCTIVE fix for the SINDy dictionary conditioning problem.

This does NOT replace your sindy_rl/pensim/env.py. It WRAPS it.

THE PROBLEM (measured, not guessed)
-----------------------------------
SINDy-RL fits dynamics AND reward as sparse dictionary models:
    x_{t+1} = Theta(x,u) @ Xi_dyn
    r_t     = Theta(x,u) @ Xi_rew
with Theta = PolynomialLibrary(degree=2, include_interaction=True).

On RAW PenSim units (pH ~ 6, Wt ~ 6e4) that degree-2 library gives:
    feature dynamic range : 6.4e+10
    cond(Theta)           : 3.3e+12
    at rew threshold 5e-1 : 133 of 136 features zeroed on MAGNITUDE ALONE

After normalizing obs/actions to [-1, 1]:
    feature dynamic range : 1.07
    cond(Theta)           : 9.7
    features zeroed       : 0 of 136

STLSQ's `threshold` and `alpha` are ABSOLUTE, not relative. They are only
meaningful when Theta columns are O(1). Your reward model is currently ~3
terms of numerical noise, and PPO is faithfully optimizing that noise.

USAGE
-----
In your PenSim yml, replace

    real_env_class: <YourPenSimEnv>
    real_env_config: {...}

with

    real_env_class: NormalizedPenSim
    real_env_config:
        wrapped_class: <YourPenSimEnv>     # the class you had before
        wrapped_config: {...}              # the config you had before
        rew_scale: 100.0

and set act_bounds/obs_bounds in the yml to normalized units (see the
accompanying dyna_pensim.yml).

Then add to sindy_rl/registry.py:
    from sindy_rl.pensim.normalize import NormalizedPenSim   # noqa: F401
'''
import numpy as np
import gymnasium
from gymnasium.spaces.box import Box


# ---------------------------------------------------------------------------
# Raw PenSim ranges, taken from YOUR box-check output.
# Verify these against pensim/env.py -- they are the only physics in this file.
# ---------------------------------------------------------------------------
ACT_NAMES = ('discharge', 'Fs', 'Foil', 'Fg', 'pres', 'Fw')
RAW_ACT_LO = np.array([0.0,    7.0,   21.0, 29.0, 0.5, 0.0])
RAW_ACT_HI = np.array([4100.0, 151.0, 36.0, 76.0, 1.2, 510.0])

OBS_NAMES = ('t', 'pH', 'T', 'Fa', 'Fb', 'Fc', 'Fh', 'Wt', 'DO2')
RAW_OBS_LO = np.array([0.0,   0.0,    118.990, 0.0,    0.0,   0.0,    0.0,      25003.258,  0.0])
RAW_OBS_HI = np.array([552.0, 16.105, 725.683, 13.717, 540.0, 3600.0, 1892.079, 253840.109, 47.899])


def to_unit(x, lo, hi):
    '''raw -> [-1, 1]'''
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return 2.0 * (np.asarray(x, float) - lo) / rng - 1.0


def from_unit(z, lo, hi):
    '''[-1, 1] -> raw (clipped at the box)'''
    return lo + (np.clip(np.asarray(z, float), -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)


def _reset_out(res):
    return res[0] if isinstance(res, tuple) else res


def _step_out(res):
    '''normalize gym/gymnasium step signatures -> (obs, rew, done, info)'''
    if len(res) == 5:
        obs, rew, term, trunc, info = res
        return obs, rew, bool(term or trunc), info
    return res


class NormalizedPenSim(gymnasium.Env):
    '''
    Wraps ANY PenSim env exposing reset()/step(raw_action).
    Presents obs and actions to SINDy-RL in [-1, 1]; converts back at the
    boundary with the real simulator. Raw engineering units never reach
    the polynomial library.
    '''

    def __init__(self, config=None):
        config = config or {}
        self.config = config

        self.act_lo = np.array(config.get('raw_act_lo', RAW_ACT_LO), float)
        self.act_hi = np.array(config.get('raw_act_hi', RAW_ACT_HI), float)
        self.obs_lo = np.array(config.get('raw_obs_lo', RAW_OBS_LO), float)
        self.obs_hi = np.array(config.get('raw_obs_hi', RAW_OBS_HI), float)

        self.act_dim = len(self.act_lo)
        self.obs_dim = len(self.obs_lo)
        self.rew_scale = float(config.get('rew_scale', 1.0))
        self.max_episode_steps = int(config.get('max_episode_steps', 552))

        self.action_space = Box(-1.0, 1.0, (self.act_dim,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, (self.obs_dim,),
                                     dtype=np.float32)

        self.n_episode_steps = 0
        self.raw_obs = None
        self.raw_env = self._build_wrapped(config)

    def _build_wrapped(self, config):
        '''Instantiate YOUR existing PenSim env, untouched.'''
        cls = config['wrapped_class']
        if isinstance(cls, str):
            from sindy_rl import registry
            cls = getattr(registry, cls)
        return cls(config.get('wrapped_config', {}))

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed=None, options=None):
        self.n_episode_steps = 0
        self.raw_obs = np.asarray(_reset_out(self.raw_env.reset()), float)
        return self.scale_obs(self.raw_obs).astype(np.float32), {}

    def step(self, action):
        raw_action = self.unscale_action(action)
        raw_obs, raw_rew, done, info = _step_out(self.raw_env.step(raw_action))

        self.raw_obs = np.asarray(raw_obs, float)
        self.n_episode_steps += 1

        info = dict(info or {})
        info.update(raw_obs=self.raw_obs, raw_action=raw_action,
                    raw_reward=float(raw_rew))

        return (self.scale_obs(self.raw_obs).astype(np.float32),
                float(raw_rew) * self.rew_scale,
                bool(done),
                self.n_episode_steps >= self.max_episode_steps,
                info)

    # -------------------------------------------------------------- scaling
    def scale_obs(self, raw):
        return to_unit(raw, self.obs_lo, self.obs_hi)

    def unscale_obs(self, norm):
        return from_unit(norm, self.obs_lo, self.obs_hi)

    def scale_action(self, raw):
        '''Use this to convert EXPERT / off-policy buffers into [-1,1].'''
        return to_unit(raw, self.act_lo, self.act_hi)

    def unscale_action(self, norm):
        return from_unit(norm, self.act_lo, self.act_hi)


def convert_buffer(in_path, out_path, rew_scale=100.0):
    '''
    Your existing off-policy .pkl is in RAW units. Feeding it to a normalized
    model is worse than having no buffer at all. Convert it once.
    '''
    import pickle
    with open(in_path, 'rb') as f:
        buf = pickle.load(f)

    buf['x'] = [to_unit(t, RAW_OBS_LO, RAW_OBS_HI) for t in buf['x']]
    buf['u'] = [to_unit(t, RAW_ACT_LO, RAW_ACT_HI) for t in buf['u']]
    if 'rew' in buf:
        buf['rew'] = [np.asarray(t, float) * rew_scale for t in buf['rew']]

    with open(out_path, 'wb') as f:
        pickle.dump(buf, f)
    print(f'normalized buffer -> {out_path}')
