from gymnasium.envs.registration import register

from rlinf.envs.realworld.dosw1.tasks.pick_and_place import (
    PickAndPlaceEnv as PickAndPlaceEnv,
)

register(
    id="DOSW1PickAndPlaceEnv-v1",
    entry_point="rlinf.envs.realworld.dosw1.tasks:PickAndPlaceEnv",
)

__all__ = ["PickAndPlaceEnv"]
