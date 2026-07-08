"""
Static configuration for the PenSim (penicillin fermentation) world model.

These values mirror the defaults exposed by ``smpl.envs.pensimenv.PenSimEnvGym``
(observation_dim=9, action_dim=6) and its documented max/min observation/action
ranges. They are kept here so every stage of the pipeline (data collection,
world-model training, later BCNP / policy stages) shares one source of truth.

Nothing in this module imports the environment, so it is safe to import anywhere.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Dimensions
# ----------------------------------------------------------------------------
OBS_DIM = 9      # state variables predicted by the GP world model
ACTION_DIM = 6   # manipulated inputs (recipe setpoints)

# ----------------------------------------------------------------------------
# Observation bounds (from PenSimEnvGym defaults).
#
# The env documents 7 of the 9 channels via ``get_observation_data_reformed``
# (Temperature, Acid flow rate, Base flow rate, Cooling water, Heating water,
# Vessel Weight, Dissolved oxygen concentration). The remaining channels are not
# labelled in the public docs, so we keep generic names for the full vector and
# do NOT rely on the labels anywhere in the pipeline.
# ----------------------------------------------------------------------------
MAX_OBS = np.array(
    [552.0, 16.10523, 725.6828, 13.717274, 540.0, 3600.0002, 1892.07874, 253840.11, 47.898834],
    dtype=np.float64,
)
MIN_OBS = np.array(
    [0.0, 0.0, 118.98977, 0.0, 0.0, 0.0, 0.0, 25003.258, 0.0],
    dtype=np.float64,
)
OBS_NAMES = [f"obs_{i}" for i in range(OBS_DIM)]

# ----------------------------------------------------------------------------
# Action bounds (from PenSimEnvGym defaults).
# Order matches the 6-vector accepted by env.step(...) in the demo.
# ----------------------------------------------------------------------------
MAX_ACTION = np.array([4100.0, 151.0, 36.0, 76.0, 1.2, 510.0], dtype=np.float64)
MIN_ACTION = np.array([0.0, 7.0, 21.0, 29.0, 0.5, 0.0], dtype=np.float64)
ACTION_NAMES = [f"act_{i}" for i in range(ACTION_DIM)]

# A physically-valid baseline action (the one used in the SMPL demo). We collect
# *sub-optimal* trajectories by perturbing around this operating point.
NOMINAL_ACTION = np.array([3000.0, 50.0, 30.0, 50.0, 0.9, 200.0], dtype=np.float64)

# Episode length: 230 h at a 12-min step  ->  1150 steps  (== env max_steps).
STEPS_PER_EPISODE = 1150

# Physical timing of the PenSim batch (SMPL default: 230 h, 12-min step).
# Breakpoints are defined in HOURS (physically portable across sample-rate /
# duration choices) and converted to step indices with STEP_SIZE_HOURS.
EPISODE_HOURS = 230.0
STEP_SIZE_HOURS = 0.2                       # 12 min per step
# Domain phase transitions (biomass accumulation -> penicillin production, etc.).
# 40 h -> step 200, 150 h -> step 750, for the default 0.2 h step.
PHASE_BREAKPOINTS_HOURS = [40.0, 150.0]


def breakpoints_to_steps(breakpoints_hours=None, step_size_hours=None):
    """Convert breakpoints in hours to step indices for this batch config."""
    bp = PHASE_BREAKPOINTS_HOURS if breakpoints_hours is None else breakpoints_hours
    ss = STEP_SIZE_HOURS if step_size_hours is None else step_size_hours
    return [int(round(h / ss)) for h in bp]


def clip_action(a: np.ndarray) -> np.ndarray:
    """Clip a raw action to the valid [MIN_ACTION, MAX_ACTION] box."""
    return np.clip(np.asarray(a, dtype=np.float64), MIN_ACTION, MAX_ACTION)
