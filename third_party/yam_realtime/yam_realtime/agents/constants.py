from typing import Dict, Union

from dm_env.specs import Array

ActionSpec = Union[Array, Dict[str, "ActionSpec"]]
"""Action specification for the agent/robot. It also includes the action space for the gripper."""
