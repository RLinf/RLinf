添加新环境
============================

本文档提供了在 RLinf 框架中添加新环境的详细说明。  
RLinf 支持多种强化学习环境，包括机器人操作（例如 ManiSkill3、LIBERO）等。  

RLinf 的环境系统由以下组件构成：

- **基础环境类**：继承自 ``gym.Env`` 的具体实现。  
- **环境封装器（Wrappers）**：提供额外功能的扩展封装器。  
- **任务变体**：实现特定任务或场景。  

1. 创建基础环境类
-----------------------------------

1.1 继承 ``gym.Env``
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

           # 初始化环境相关参数
           self.num_envs = num_envs
           self.group_size = self.cfg.group_size
           self.num_group = self.num_envs // self.group_size

           # 初始化环境内部
           self._init_environment()
           self._init_reset_state_ids()

       def _init_environment(self):
           """初始化具体的环境实例。"""
           # 根据环境类型初始化
           pass

       def _init_reset_state_ids(self):
           """初始化重置状态 ID 和随机数发生器。"""
           self._generator = torch.Generator()
           self._generator.manual_seed(self.seed)
           # 设置重置状态逻辑
           pass

       def reset(self, options={}):
           """重置环境。"""
           # 实现环境重置逻辑
           obs = self._get_observation()
           return obs, {}

       def step(self, actions):
           """执行动作。"""
           # 实现动作执行逻辑
           obs = self._get_observation()
           reward = self._calculate_reward()
           terminated = self._check_termination()
           truncated = self._check_truncation()
           info = self._get_info()
           return obs, reward, terminated, truncated, info

       def _get_observation(self):
           """获取观测。"""
           # 实现观测获取逻辑
           pass

       def _calculate_reward(self):
           """计算奖励。"""
           # 实现奖励计算逻辑
           pass

       def _check_termination(self):
           """检查终止条件。"""
           # 实现终止条件逻辑
           pass

       def _check_truncation(self):
           """检查截断条件。"""
           # 实现截断条件逻辑
           pass

       def _get_info(self):
           """获取信息字典。"""
           # 实现信息获取逻辑
           pass

1.2 实现必要的属性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   @property
   def total_num_group_envs(self):
       """环境组的总数量。"""
       # 根据你的环境实现
       pass

   @property
   def num_envs(self):
       """向量化环境的数量。"""
       return self.num_envs

   @property
   def device(self):
       """当前使用的设备。"""
       return self.env.unwrapped.device

2. 实现环境的 Offload 支持（可选）
----------------------------------------------------------------------

如果需要支持保存/恢复环境状态，可以继承 ``EnvOffloadMixin``：

.. code-block:: python

   from rlinf.envs.env_offload_wrapper import EnvOffloadMixin
   import io
   import torch

   class YourCustomEnv(gym.Env, EnvOffloadMixin):
       def get_state(self) -> bytes:
           """序列化环境状态为字节。"""
           state = {
               "env_state": self.env.get_state(),
               "rng_state": self._generator.get_state(),
               # 根据需要添加其他状态
           }
           buffer = io.BytesIO()
           torch.save(state, buffer)
           return buffer.getvalue()

       def load_state(self, state_buffer: bytes):
           """从字节恢复环境状态。"""
           buffer = io.BytesIO(state_buffer)
           state = torch.load(buffer, map_location="cpu")
           self.env.set_state(state["env_state"])
           self._generator.set_state(state["rng_state"])
           # 根据需要恢复其他状态

3. 创建环境封装器
-----------------------------------

如果实现了 offload 功能，需要创建对应的封装器：

.. code-block:: python

   # 在 env_offload_wrapper.py 中
   class YourCustomEnv(BaseYourCustomEnv, EnvOffloadMixin):
       def get_state(self) -> bytes:
           # 实现状态保存
           pass

       def load_state(self, state_buffer: bytes):
           # 实现状态恢复
           pass

4. 添加动作处理工具
-----------------------------------

在 ``action_utils.py`` 中添加动作处理函数：

.. code-block:: python

   def prepare_actions_for_your_env(
       raw_chunk_actions,
       num_action_chunks,
       action_dim,
       action_scale,
       policy,
   ):
       """为你的环境准备动作。"""
       # 实现动作处理逻辑
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
       # ... 其他环境类型
       return chunk_actions

5. 创建任务变体（可选）
-----------------------------------

如果需要特定任务变体，可以将其放在 ``envs/YOUR_ENV/tasks/variants/`` 下：

.. code-block:: python

   # envs/YOUR_ENV/tasks/variants/your_task_variant.py
   class YourTaskVariant:
       def __init__(self, config):
           self.config = config

       def setup_task(self):
           """设置任务资源和初始状态。"""
           pass

       def get_task_description(self):
           """返回任务的自然语言描述。"""
           pass

       def check_success(self, obs, action):
           """任务成功时返回 True。"""
           pass

6. 更新配置文件
-----------------------------------

添加你的环境配置：

.. code-block:: yaml

   your_env:
     env_type: "your_env"
     total_num_envs: 8
     group_size: 4
     seed: 42
     # 其他环境特定设置

7. 注册环境
-----------------------------------

在包中暴露新环境：

.. code-block:: python

   # 在 __init__.py 或相关模块中
   from .your_custom_env import YourCustomEnv

   __all__ = ["YourCustomEnv"]

测试与验证
-----------------------------------

.. code-block:: python

   import numpy as np

   def test_your_env():
       """对环境的基本测试。"""
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
