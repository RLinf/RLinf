基于规则的 Reward
=================

无需修改 RLinf 的 runner 或 worker，即可为推理与 Agent 任务注册确定性的规则 Reward 打分类。配置中 ``reward.use_reward_model`` 为 ``False`` 时，``RewardWorker`` 会通过 ``reward_registry`` 按 ``reward.reward_type`` 加载对应类。

扩展点速览
----------

.. list-table::
   :header-rows: 1
   :widths: 22 28 24 26

   * - 扩展组件
     - 注册方式
     - 配置字段
     - 参考源码
   * - 规则 Reward
     - ``register_reward(name, cls)``
     - ``reward.reward_type``
     - ``rlinf/algorithms/rewards/__init__.py``

1. 实现并注册打分类
-------------------

在 ``rlinf/algorithms/rewards/<name>/__init__.py``\ （或外部扩展包）中实现 ``__init__(self, config)`` 与 ``get_reward(self, completions, answers, **kwargs)``，并通过 ``register_reward`` 注册：

.. code-block:: python

   from omegaconf import DictConfig

   from rlinf.algorithms.rewards import register_reward


   class ExactMatchReward:
       def __init__(self, config: DictConfig):
           self.scale = float(config.get("reward_scale", 1.0))

       def get_reward(
           self,
           completions: list[str],
           answers: list[str],
           **kwargs,
       ) -> list[float]:
           return [
               self.scale if pred.strip() == str(ans).strip() else 0.0
               for pred, ans in zip(completions, answers)
           ]


   register_reward("exact_match", ExactMatchReward)

调用链路说明：

- ``__init__(self, config)`` 接收 Hydra 配置中的 ``cfg.reward`` 节点。
- ``get_reward(self, completions, answers, **kwargs)`` 按顺序为每条回复返回一个浮点分值；如需同时获取输入 prompt，在 YAML 中设置 ``reward.use_prompt: True`` 即可通过 ``kwargs["prompts"]`` 读取。
- ``register_reward("exact_match", ExactMatchReward)`` 将类写入 ``reward_registry``，``RewardWorker`` 启动时会通过 ``get_rule_based_reward_class("exact_match")`` 实例化它。

2. 在 YAML 中启用并本地验证
---------------------------

在推理或 Agent YAML 中启用：

.. code-block:: yaml

   reward:
     use_reward_model: False
     reward_type: exact_match
     reward_scale: 1.0

在本地 CPU 上验证注册与加载（无需启动 Ray）：

.. code-block:: bash

   python -c '
   from omegaconf import OmegaConf
   from rlinf.algorithms.rewards import get_rule_based_reward_class

   reward_cls = get_rule_based_reward_class("math")
   scorer = reward_cls(OmegaConf.create({"reward_scale": 1.0}))
   print("loaded:", type(scorer).__name__)
   '

3. 打包与启动
-------------

.. tab-set::

   .. tab-item:: 主仓库内注册（In-Tree）
      :sync: ext-in-tree

      在 ``rlinf/algorithms/rewards/__init__.py`` 中导入你的模块，确保每个 worker 导入时自动生效，启动时指定 ``reward.reward_type`` 即可。

   .. tab-item:: 外部仓库注册（Out-of-Tree）
      :sync: ext-out-of-tree

      在外部扩展包的 ``register()`` 中导入你的 Reward 模块，启动前导出 ``RLINF_EXT_MODULE``\ （见 :ref:`out-of-tree-extensions`\ ）：

      .. code-block:: python

         # my_pkg/rlinf_ext.py
         def register():
             import my_pkg.my_rewards  # 执行 register_reward(...)

      .. code-block:: bash

         export RLINF_EXT_MODULE=my_pkg.rlinf_ext
