Simulator Env
=============

Wrap a vectorized simulator as a ``gym.Env``, register its environment type, map policy actions to its action space, and verify the interface locally on CPU. Once registered, any embodied RL config can select the environment through ``env.train.env_type`` or ``env.eval.env_type``.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Item
     - Details
   * - Files to create/edit
     - ``rlinf/envs/sim/<name>/``, ``rlinf/envs/__init__.py``, ``rlinf/envs/action_utils.py``, ``examples/embodiment/config/env/<name>.yaml``
   * - Extension point
     - ``SupportedEnvType`` and ``get_env_cls()`` in ``rlinf/envs/__init__.py``
   * - Required interface
     - ``gym.Env`` with ``reset``, ``step``, ``num_envs``, ``total_num_group_envs``, and ``device``
   * - Local CPU verification
     - Run reset and step on the pure-PyTorch example environment directly on CPU without launching Ray

1. Implement a Vectorized Environment
-------------------------------------

Place your simulator package under ``rlinf/envs/sim/<name>/`` with an ``__init__.py``. Every simulator wrapper inherits from ``gym.Env`` and manages ``num_envs`` parallel instances per worker stage.

The example below implements a complete, runnable 2D point-reaching environment (PointReachEnv) using only ``torch`` and ``gymnasium`` so you can inspect the exact tensor shapes and run it on CPU. Save it as ``rlinf/envs/sim/point_reach/point_reach_env.py``:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   import torch


   class PointReachEnv(gym.Env):
       """Vectorized 2D point-reaching environment running on CPU."""

       def __init__(
           self,
           cfg,
           num_envs: int,
           seed_offset: int = 0,
           total_num_processes: int = 1,
           worker_info=None,
       ):
           self.cfg = cfg
           self._num_envs = num_envs
           self.seed = cfg.seed + seed_offset
           self.total_num_processes = total_num_processes
           self.worker_info = worker_info
           self.group_size = cfg.group_size
           self.num_group = self._num_envs // self.group_size
           self.max_episode_steps = cfg.max_episode_steps
           self._device = torch.device("cpu")
           self._generator = torch.Generator(device=self._device).manual_seed(self.seed)

           self.action_space = gym.spaces.Box(
               low=-1.0, high=1.0, shape=(2,), dtype=np.float32
           )
           self.observation_space = gym.spaces.Dict(
               {
                   "states": gym.spaces.Box(
                       low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                   )
               }
           )
           self.pos = torch.zeros(self._num_envs, 2, device=self._device)
           self.target = torch.zeros(self._num_envs, 2, device=self._device)
           self.elapsed_steps = torch.zeros(
               self._num_envs, dtype=torch.long, device=self._device
           )

       @property
       def num_envs(self) -> int:
           return self._num_envs

       @property
       def total_num_group_envs(self) -> int:
           return self.num_group

       @property
       def device(self) -> torch.device:
           return self._device

       def _get_obs(self) -> dict[str, torch.Tensor]:
           return {"states": torch.cat([self.pos, self.target], dim=-1)}

       def reset(self, *, seed: int | None = None, options: dict | None = None):
           if seed is not None:
               self._generator.manual_seed(seed)
           self.pos.zero_()
           self.target = (
               torch.rand(
                   self._num_envs, 2, generator=self._generator, device=self._device
               )
               * 2.0
               - 1.0
           )
           self.elapsed_steps.zero_()
           return self._get_obs(), {}

       def step(self, actions: torch.Tensor | np.ndarray):
           actions = torch.as_tensor(
               actions, dtype=torch.float32, device=self._device
           ).clamp(-1.0, 1.0)
           self.pos = self.pos + 0.1 * actions
           self.elapsed_steps += 1

           dist = torch.linalg.norm(self.pos - self.target, dim=-1)
           reward = -dist
           terminated = dist < 0.05
           truncated = self.elapsed_steps >= self.max_episode_steps
           info = {"distance": dist, "success": terminated}
           return self._get_obs(), reward, terminated, truncated, info

How ``EnvWorker`` interacts with this class, in call order:

- ``__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)``: ``EnvWorker`` passes the Hydra env config, the number of parallel environments for this pipeline stage, a unique ``seed_offset`` per stage, ``total_num_processes``, and ``worker_info``. Use ``cfg.seed + seed_offset`` so ranks do not sample identical rollouts.
- ``reset(seed=None, options=None)``: resets all ``num_envs`` instances and returns ``(obs, info)`` where every tensor in ``obs`` has leading batch dimension ``num_envs`` (here ``states`` has shape ``[num_envs, 4]``).
- ``step(actions)``: accepts actions of shape ``[num_envs, action_dim]`` and returns ``(obs, reward, terminated, truncated, info)`` where ``reward`` is float tensor ``[num_envs]`` and ``terminated`` / ``truncated`` are boolean tensors ``[num_envs]``.
- ``num_envs``, ``total_num_group_envs``, and ``device``: properties read by ``EnvWorker`` and wrappers for batching, GRPO group indexing, and device placement.

.. note::

   Embodied policies that output multi-step action chunks call ``chunk_step(chunk_actions)`` on the environment (see ``rlinf/envs/sim/maniskill/maniskill_env.py`` for a full ``chunk_step`` and ``auto_reset`` implementation).

2. Add Offload Support (Optional)
---------------------------------

If your GPU simulator shares devices with actor or rollout workers, support offloading by serializing simulator and RNG state in ``get_state()`` and restoring it in ``load_state()`` (see ``EnvOffloadMixin`` and ``ManiskillOffloadEnv`` in ``rlinf/envs/sim/maniskill/maniskill_offload_env.py``):

.. code-block:: python

   import io
   import torch


   class PointReachOffloadEnv(PointReachEnv):
       def get_state(self) -> bytes:
           state = {
               "pos": self.pos.cpu(),
               "target": self.target.cpu(),
               "elapsed_steps": self.elapsed_steps.cpu(),
               "rng_state": self._generator.get_state(),
           }
           buffer = io.BytesIO()
           torch.save(state, buffer)
           return buffer.getvalue()

       def load_state(self, state_buffer: bytes) -> None:
           state = torch.load(
               io.BytesIO(state_buffer), map_location="cpu", weights_only=True
           )
           self.pos = state["pos"].to(self._device)
           self.target = state["target"].to(self._device)
           self.elapsed_steps = state["elapsed_steps"].to(self._device)
           self._generator.set_state(state["rng_state"])

3. Register the Environment and Format Actions
----------------------------------------------

Register the environment in ``rlinf/envs/__init__.py`` by adding a ``SupportedEnvType`` member and a lazy-import branch inside ``get_env_cls()``. Keep the import inside the ``elif`` branch so optional simulator packages are loaded only when selected:

.. code-block:: python

   class SupportedEnvType(Enum):
       ...
       POINT_REACH = "point_reach"


   def get_env_cls(env_type: str, env_cfg=None):
       env_type = SupportedEnvType(env_type)
       ...
       elif env_type == SupportedEnvType.POINT_REACH:
           from rlinf.envs.sim.point_reach.point_reach_env import PointReachEnv

           return PointReachEnv

Next, check ``prepare_actions()`` in ``rlinf/envs/action_utils.py``. If your simulator consumes the policy's action tensor directly, the default ``else`` branch passes ``raw_chunk_actions`` through unchanged. If your simulator needs rescaling, gripper binarization, or coordinate remapping, add a branch for your ``SupportedEnvType``:

.. code-block:: python

   def prepare_actions_for_point_reach(raw_chunk_actions, action_scale: float):
       return np.clip(raw_chunk_actions * action_scale, -1.0, 1.0)


   def prepare_actions(
       raw_chunk_actions,
       env_type: str,
       model_type: str,
       num_action_chunks,
       action_dim,
       action_scale: float = 1.0,
       policy: str = "widowx_bridge",
       wm_env_type=None,
       env_cfg=None,
   ):
       ...
       elif env_type == SupportedEnvType.POINT_REACH:
           chunk_actions = prepare_actions_for_point_reach(
               raw_chunk_actions=raw_chunk_actions,
               action_scale=action_scale,
           )
       ...
       return chunk_actions

4. Add a Hydra Configuration Preset
-----------------------------------

Create a Hydra preset at ``examples/embodiment/config/env/point_reach.yaml`` with static defaults:

.. code-block:: yaml

   env_type: point_reach
   total_num_envs: 8
   group_size: 4
   max_episode_steps: 50
   seed: 42
   auto_reset: true
   ignore_terminations: false

   video_cfg:
     save_video: false
     video_base_dir: ${runner.logger.log_path}/video/train

5. Verify Locally on CPU
------------------------

Before launching a full Ray training job, verify ``reset``, ``step``, tensor shapes, dtypes, and required properties directly on CPU:

.. code-block:: python

   import torch
   from omegaconf import OmegaConf

   from rlinf.envs.sim.point_reach.point_reach_env import PointReachEnv

   cfg = OmegaConf.create(
       {
           "env_type": "point_reach",
           "total_num_envs": 8,
           "group_size": 4,
           "max_episode_steps": 50,
           "seed": 42,
       }
   )
   env = PointReachEnv(cfg, num_envs=cfg.total_num_envs)

   obs, info = env.reset(seed=42)
   assert obs["states"].shape == (8, 4) and info == {}

   actions = torch.zeros(env.num_envs, 2)
   next_obs, reward, terminated, truncated, step_info = env.step(actions)
   assert next_obs["states"].shape == (8, 4)
   assert reward.shape == (8,) and reward.dtype == torch.float32
   assert terminated.shape == (8,) and terminated.dtype == torch.bool
   assert truncated.shape == (8,) and truncated.dtype == torch.bool
   assert "distance" in step_info and "success" in step_info
   assert env.total_num_group_envs == 2 and env.device == torch.device("cpu")
   print("PointReachEnv check passed:", env.num_envs, env.device)
