from gymnasium.envs.registration import register
from rlinf.envs.physical.franka.tasks.peg_insertion_env import PegInsertionEnv
from rlinf.envs.physical.franka.tasks.drone_peg_insertion_env import DronePegInsertionEnv

register(
    id="PegInsertionEnv-v1", 
    entry_point="rlinf.envs.physical.franka.tasks:PegInsertionEnv", 
)

register(
    id="DronePegInsertionEnv-v1", 
    entry_point="rlinf.envs.physical.franka.tasks:DronePegInsertionEnv", 
)