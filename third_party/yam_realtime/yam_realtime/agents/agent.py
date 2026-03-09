from dataclasses import dataclass
from typing import Any, Dict, Protocol

from yam_realtime.agents.constants import ActionSpec
from yam_realtime.utils.portal_utils import remote


class Agent(Protocol):
    use_joint_state_as_action: bool = False

    def act(self, obs: Dict[str, Any]) -> Any:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError

    @remote()
    def action_spec(self) -> ActionSpec:
        """Check if the agent is compatible with the environment.

        Args:
            action_spec: dictionary of action specification.
        """
        raise NotImplementedError


@dataclass
class PolicyAgent(Agent):
    use_joint_state_as_action: bool = False

    def act(self, obs: Dict[str, Any]) -> Any:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError

    @remote()
    def action_spec(self) -> ActionSpec:
        """Check if the agent is compatible with the environment.

        Args:
            action_spec: dictionary of action specification.
        """
        raise NotImplementedError
