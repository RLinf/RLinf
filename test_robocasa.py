from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts
import numpy as np

# choose random task
# env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
env_name = 'PnPCounterToCab'

env = create_env(
    env_name=env_name,
    render_onscreen=False,
    seed=0, # set seed=None to run unseeded
)

# run rollouts with random actions and save video
info = run_random_rollouts(
    env, num_rollouts=1, num_steps=100, video_path="test.mp4"
)
# print(info)

# # reset environment
# obs = env.reset()
# # import pdb; pdb.set_trace()

# num_steps = 100

# for step in range(num_steps):
#     # sample random action
#     action = env.action_spec.sample()

#     # take one step
#     obs, reward, terminated, truncated, info = env.step(action)

#     # print debug information
#     print(f"\n--- Step {step} ---")
#     print("Action:       ", action)
#     print("Observation:  ", obs)
#     print("Reward:       ", reward)
#     print("Terminated:   ", terminated)
#     print("Truncated:    ", truncated)
#     print("Info keys:    ", list(info.keys()))

#     # exit rollout if env ended
#     if terminated or truncated:
#         print("\nEnvironment finished, resetting...\n")
#         obs = env.reset()

# env.close()