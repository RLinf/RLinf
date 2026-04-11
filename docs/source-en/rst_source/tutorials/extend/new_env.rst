Adding New Environment 
============================

This document provides detailed instructions on how to add new environments to the RLinf framework.  
RLinf supports various reinforcement learning environments, including robotic manipulation (e.g., ManiSkill3, LIBERO) and others.

The RLinf environment system consists of the following components:

- **Base Environment Classes**: Concrete implementations inheriting from ``gym.Env``.
- **Environment Wrappers**: Add-on wrappers that provide extra functionality.
- **Task Variants**: Implementations of specific tasks or scenarios.


1. Create Base Environment Class
-----------------------------------

1.1 Inherit from ``gym.Env``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   import torch

   class YourCustomEnv(gym.Env):
       def __init__(self, cfg, rank, num_envs, ret_device="cpu"):
           self.cfg = cfg
           self.rank = rank
           self.ret_device = ret_device
           self.seed = self.cfg.seed + rank

           # Initialize environment-related parameters
           self.num_envs = num_envs
           self.group_size = self.cfg.group_size
           self.num_group = self.num_envs // self.group_size

           # Initialize environment internals
           self._init_environment()
           self._init_reset_state_ids()

       def _init_environment(self):
           """Initialize the specific environment instance."""
           # Initialize based on environment type
           pass

       def _init_reset_state_ids(self):
           """Initialize reset state IDs and RNG."""
           self._generator = torch.Generator()
           self._generator.manual_seed(self.seed)
           # Set up reset-state logic
           pass

       def reset(self, options={}):
           """Reset the environment."""
           # Implement environment reset logic
           obs = self._get_observation()
           return obs, {}

       def step(self, actions):
           """Execute actions."""
           # Implement action execution logic
           obs = self._get_observation()
           reward = self._calculate_reward()
           terminated = self._check_termination()
           truncated = self._check_truncation()
           info = self._get_info()
           return obs, reward, terminated, truncated, info

       def _get_observation(self):
           """Retrieve observation."""
           # Implement observation retrieval logic
           pass

       def _calculate_reward(self):
           """Compute reward."""
           # Implement reward calculation logic
           pass

       def _check_termination(self):
           """Check termination conditions."""
           # Implement termination condition checks
           pass

       def _check_truncation(self):
           """Check truncation conditions."""
           # Implement truncation condition checks
           pass

       def _get_info(self):
           """Retrieve info dict."""
           # Implement information retrieval logic
           pass

1.2 Implement Required Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   @property
   def total_num_group_envs(self):
       """Total number of environment groups."""
       # Implement based on your environment
       pass

   @property
   def num_envs(self):
       """Number of vectorized environments."""
       return self.num_envs

   @property
   def device(self):
       """Active device."""
       return self.env.unwrapped.device

2. Implement Environment Offload Support (Optional)
----------------------------------------------------------------------

If you need to support saving/restoring environment state, inherit from ``EnvOffloadMixin``:

.. code-block:: python

   from rlinf.envs.env_offload_wrapper import EnvOffloadMixin
   import io
   import torch

   class YourCustomEnv(gym.Env, EnvOffloadMixin):
       def get_state(self) -> bytes:
           """Serialize environment state to bytes."""
           state = {
               "env_state": self.env.get_state(),
               "rng_state": self._generator.get_state(),
               # Add other states as needed
           }
           buffer = io.BytesIO()
           torch.save(state, buffer)
           return buffer.getvalue()

       def load_state(self, state_buffer: bytes):
           """Restore environment state from bytes."""
           buffer = io.BytesIO(state_buffer)
           state = torch.load(buffer, map_location="cpu")
           self.env.set_state(state["env_state"])
           self._generator.set_state(state["rng_state"])
           # Restore other states as needed

3. Create Environment Wrapper
-----------------------------------

If you implement offload functionality, create a corresponding wrapper:

.. code-block:: python

   # In env_offload_wrapper.py
   class YourCustomEnv(BaseYourCustomEnv, EnvOffloadMixin):
       def get_state(self) -> bytes:
           # Implement state saving
           pass

       def load_state(self, state_buffer: bytes):
           # Implement state restoration
           pass

4. Add Action Processing Tools
-----------------------------------

Add action processing utilities in ``action_utils.py``:

.. code-block:: python

   def prepare_actions_for_your_env(
       raw_chunk_actions,
       num_action_chunks,
       action_dim,
       action_scale,
       policy,
   ):
       """Prepare actions for your environment."""
       # Implement action processing logic
       pass

   def prepare_actions(
       env_type,
       raw_chunk_actions,
       num_action_chunks,
       action_dim,
       action_scale: float = 1.0,
       policy: str = "default",
   ):
       if env_type == "your_env":
           chunk_actions = prepare_actions_for_your_env(
               raw_chunk_actions=raw_chunk_actions,
               num_action_chunks=num_action_chunks,
               action_dim=action_dim,
               action_scale=action_scale,
               policy=policy,
           )
       # ... other environment types
       return chunk_actions

5. Create Task Variants (Optional)
-----------------------------------

If you require specific task variants, place them under ``envs/YOUR_ENV/tasks/variants/``:

.. code-block:: python

   # envs/YOUR_ENV/tasks/variants/your_task_variant.py
   class YourTaskVariant:
       def __init__(self, config):
           self.config = config

       def setup_task(self):
           """Set up task assets and initial state."""
           pass

       def get_task_description(self):
           """Return a natural-language task description."""
           pass

       def check_success(self, obs, action):
           """Return True if the task is successful."""
           pass

6. Update Configuration Files
-----------------------------------

Add your environment configuration:

.. code-block:: yaml

   your_env:
     env_type: "your_env"
     total_num_envs: 8
     group_size: 4
     seed: 42
     # Other environment-specific settings

7. Register Environment
-----------------------------------

Expose the new environment in the package:

.. code-block:: python

   # In __init__.py or the relevant module
   from .your_custom_env import YourCustomEnv

   __all__ = ["YourCustomEnv"]

Testing and Validation
-----------------------------------

.. code-block:: python

   import numpy as np

   def test_your_env():
       """Basic smoke test for your environment."""
       cfg = get_test_config()
       env = YourCustomEnv(cfg, rank=0)

       # Reset
       obs, info = env.reset()
       assert obs is not None

       # Step
       action = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(action)
       assert obs is not None
       assert isinstance(reward, (float, np.ndarray))

