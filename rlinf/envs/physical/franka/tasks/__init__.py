from gymnasium.envs.registration import register
from rlinf.envs.physical.franka.tasks.peg_insertion_env import PegInsertionEnv

register(
    id="PegInsertionEnv-v1", 
    entry_point="rlinf.envs.physical.franka.tasks:PegInsertionEnv", 
)