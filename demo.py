from smpl.envs.pensimenv import PenSimEnvGym
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from pensimpy.data.constants import FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE, \
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE
import numpy as np

# Build the default recipe combo (baseline operating procedure)
recipe_dict = {FS: Recipe(FS_DEFAULT_PROFILE, FS),
               FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
               FG: Recipe(FG_DEFAULT_PROFILE, FG),
               PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
               DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
               WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
               PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)}
recipe_combo = RecipeCombo(recipe_dict=recipe_dict)

# Custom reward: raw penicillin yield per step
def raw_yield_reward(previous_observation, action, current_observation, reward=None):
    return reward  # yield_per_run is already computed by pensimpy and passed as reward

# Create environment
env = PenSimEnvGym(
    recipe_combo=recipe_combo,
    reward_function=raw_yield_reward,
    dense_reward=True
)

obs = env.reset()
print("Initial observation:", obs)

for step in range(5):
    action = np.array([3000.0, 50.0, 30.0, 50.0, 0.9, 200.0])
    obs, reward, done, info = env.step(action)
    print(f"Step {step+1}: Yield={reward:.4f}, Done={done}")
    if done:
        break
