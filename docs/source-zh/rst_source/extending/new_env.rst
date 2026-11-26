仿真 Env
========

将向量化仿真器封装为 ``gym.Env``、注册环境类型、完成动作格式映射，并在本地 CPU 上验证接口契约。注册完成后，任意具身强化学习配置都可以通过 ``env.train.env_type`` 或 ``env.eval.env_type`` 直接选用该环境。

接入速览
--------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 项目
     - 说明
   * - 新增或修改的文件
     - ``rlinf/envs/sim/<name>/``、``rlinf/envs/__init__.py``、``rlinf/envs/action_utils.py``、``examples/embodiment/config/env/<name>.yaml``
   * - 扩展点
     - ``rlinf/envs/__init__.py`` 中的 ``SupportedEnvType`` 与 ``get_env_cls()``
   * - 必选接口
     - 继承 ``gym.Env``，提供 ``reset``、``step``、``num_envs``、``total_num_group_envs`` 与 ``device``
   * - 本地 CPU 验证
     - 无需启动 Ray，直接在 CPU 上运行纯 PyTorch 示例环境的 ``reset`` 与 ``step`` 检查

1. 实现向量化环境
-----------------

在 ``rlinf/envs/sim/<name>/`` 下创建仿真器包并添加 ``__init__.py``。每个仿真环境类都继承自 ``gym.Env``，并在单个 worker stage 内管理 ``num_envs`` 个并行实例。

下面的示例仅依赖 ``torch`` 和 ``gymnasium`` 实现了一个可直接运行的二维点位到达环境（PointReachEnv），完整展示了并行张量形状，可直接在 CPU 上运行验证。将其保存为 ``rlinf/envs/sim/point_reach/point_reach_env.py``：

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   import torch


   class PointReachEnv(gym.Env):
       """运行在 CPU 上的向量化二维点位到达环境。"""

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

``EnvWorker`` 按调用顺序依赖以下接口：

- ``__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)``：``EnvWorker`` 传入 Hydra 环境配置、当前流水线 stage 的并行环境数、各 stage 独立的 ``seed_offset``、总进程数 ``total_num_processes`` 以及 ``worker_info``。用 ``cfg.seed + seed_offset`` 初始化随机种子，避免不同 rank 采出完全相同的轨迹。
- ``reset(seed=None, options=None)``：重置全部 ``num_envs`` 个实例并返回 ``(obs, info)``，``obs`` 中每个张量的首维均为 ``num_envs``\ （本例中 ``states`` 的形状为 ``[num_envs, 4]``\ ）。
- ``step(actions)``：接收形状为 ``[num_envs, action_dim]`` 的动作，返回 ``(obs, reward, terminated, truncated, info)``，其中 ``reward`` 为形状 ``[num_envs]`` 的浮点张量，``terminated`` 与 ``truncated`` 为形状 ``[num_envs]`` 的布尔张量。
- ``num_envs``、``total_num_group_envs`` 与 ``device``：供 ``EnvWorker`` 及封装器读取，用于批处理、GRPO 分组索引和设备放置。

.. note::

   输出多步动作块的具身策略会调用环境的 ``chunk_step(chunk_actions)``\ （完整实现和 ``auto_reset`` 处理可参考 ``rlinf/envs/sim/maniskill/maniskill_env.py``\ ）。

2. 支持状态卸载（可选）
-----------------------

如果 GPU 仿真器需要与 actor 或 rollout worker 共享显存，可以实现 ``get_state()`` 和 ``load_state()`` 将仿真状态与随机数状态序列化到 CPU 内存并按需恢复（完整代理模式见 ``rlinf/envs/sim/maniskill/maniskill_offload_env.py`` 中的 ``EnvOffloadMixin`` 和 ``ManiskillOffloadEnv``\ ）：

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

3. 注册环境并适配动作格式
-------------------------

在 ``rlinf/envs/__init__.py`` 的 ``SupportedEnvType`` 枚举中新增成员，并在 ``get_env_cls()`` 中补充延迟导入分支。将导入放在 ``elif`` 分支内部，确保未选中该环境时不会加载可选仿真器依赖：

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

接着检查 ``rlinf/envs/action_utils.py`` 中的 ``prepare_actions()``。如果仿真器可以直接消费策略输出的动作张量，默认的 ``else`` 分支会原样透传 ``raw_chunk_actions``；如果需要缩放、夹爪二值化或坐标映射，则按对应的 ``SupportedEnvType`` 新增处理分支：

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

4. 添加 Hydra 配置预设
----------------------

在 ``examples/embodiment/config/env/point_reach.yaml`` 中创建静态默认配置：

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

5. 在本地 CPU 上验证
--------------------

启动完整 Ray 训练前，直接在本地 CPU 上验证 ``reset``、``step``、张量形状、数据类型以及必选属性：

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
