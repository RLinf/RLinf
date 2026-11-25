新增算法
========

无需修改 RLinf 的 runner 或 worker，即可注册自定义 Advantage 函数或 Policy Loss 函数。注册完成后，任意现有 YAML 配置都能通过 ``algorithm.adv_type`` 或 ``algorithm.loss_type`` 直接选用。

扩展点速览
----------

.. list-table::
   :header-rows: 1
   :widths: 24 28 22 26

   * - 扩展组件
     - 注册方式
     - 配置字段
     - 参考源码
   * - Advantage 函数
     - ``@register_advantage(name)``
     - ``algorithm.adv_type``
     - ``rlinf/algorithms/advantages.py``
   * - Policy Loss 函数
     - ``@register_policy_loss(name)``
     - ``algorithm.loss_type``
     - ``rlinf/algorithms/losses.py``

两类注册表均按进程独立维护：主仓内新增组件在模块导入时自动注册；外部仓库通过 ``RLINF_EXT_MODULE`` 在每个 worker 中完成注册（见 :ref:`out-of-tree-extensions`\ ）。自定义 Reward 请见 :doc:`基于规则的 Reward <rule_reward>` 与 :doc:`基于模型的 Reward <reward_model>`。

1. 自定义 Advantage 函数
------------------------

在 ``rlinf/algorithms/advantages.py``\ （或外部扩展模块）中编写带 ``@register_advantage(name)`` 装饰器的函数。函数签名需保留 ``**kwargs`` 以接收公共 batch 字段，并返回 ``(advantages, returns)`` 二元组（无 critic 的算法可将 ``returns`` 设为 ``None``）。

.. code-block:: python

   import torch

   from rlinf.algorithms.registry import register_advantage


   @register_advantage("centered_reward")
   def compute_centered_advantages(
       rewards: torch.Tensor,
       loss_mask: torch.Tensor,
       normalize_advantages: bool = True,
       **kwargs,
   ) -> tuple[torch.Tensor, None]:
       scores = rewards.reshape(-1) if rewards.ndim == 2 else rewards
       advantages = scores.unsqueeze(0).expand_as(loss_mask) * loss_mask
       if normalize_advantages:
           valid = advantages[loss_mask.bool()]
           if valid.numel() > 0:
               advantages = (advantages - valid.mean()) / (valid.std() + 1e-5)
       return advantages, None

调用链路说明：

- ``@register_advantage("centered_reward")`` 按小写名称将函数写入 ``ADV_REGISTRY``。
- ``calculate_adv_and_returns(adv_type="centered_reward", task_type=..., ...)`` 会先预处理输入张量，再调用你的函数并整理为 actor worker 所需的返回格式。具身任务可直接分发到任意已注册的 ``adv_type``；如果用于推理任务，还需在 ``rlinf/algorithms/utils.py`` 的 ``preprocess_reasoning_advantages_inputs()`` 中补充对应分支。

在 YAML 或命令行中启用：

.. code-block:: yaml

   algorithm:
     adv_type: centered_reward
     normalize_advantages: True

在本地 CPU 上直接验证（无需启动 Ray）：

.. code-block:: bash

   python -c '
   import torch
   import rlinf.algorithms.advantages  # 触发主仓内置注册
   from rlinf.algorithms.registry import calculate_adv_and_returns

   out = calculate_adv_and_returns(
       adv_type="raw",
       task_type="embodied",
       reward_type="chunk_level",
       logprob_type="chunk_level",
       rewards=torch.tensor([[[1.0], [0.0]], [[0.5], [0.0]]]),
       dones=torch.zeros(3, 2, 1, dtype=torch.bool),
       loss_mask=torch.ones(2, 2, 1, dtype=torch.bool),
       group_size=2,
   )
   print("advantages shape:", out["advantages"].shape)
   '

2. 自定义 Policy Loss 函数
--------------------------

在 ``rlinf/algorithms/losses.py``\ （或外部扩展模块）中编写带 ``@register_policy_loss(name)`` 装饰器的函数。保留 ``**kwargs``，对 ``float32`` 精度做断言以保证数值稳定，并返回 ``(loss, metrics_dict)``，其中 ``loss`` 为标量张量，``metrics_dict`` 中各项为已 ``detach()`` 的标量张量或浮点数。

.. code-block:: python

   from typing import Callable, Optional

   import torch

   from rlinf.algorithms.registry import register_policy_loss
   from rlinf.utils.utils import masked_mean


   @register_policy_loss("clipped_pg")
   def compute_clipped_pg_loss(
       logprobs: torch.Tensor,
       old_logprobs: torch.Tensor,
       advantages: torch.Tensor,
       clip_ratio_low: float = 0.2,
       clip_ratio_high: float = 0.2,
       loss_mask: Optional[torch.Tensor] = None,
       loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
       **kwargs,
   ) -> tuple[torch.Tensor, dict]:
       assert logprobs.dtype == torch.float32
       assert old_logprobs.dtype == torch.float32
       assert advantages.dtype == torch.float32

       if loss_mask is None:
           loss_mask = torch.ones_like(logprobs, dtype=torch.bool)

       ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0.0)
       clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
       pg_loss = torch.maximum(-advantages * ratio, -advantages * clipped_ratio)
       loss = loss_agg_func(pg_loss, loss_mask)
       metrics = {
           "actor/policy_loss": loss.detach(),
           "actor/ratio": masked_mean(ratio, loss_mask).detach(),
       }
       return loss, metrics

调用链路说明：

- ``@register_policy_loss("clipped_pg")`` 将 Loss 函数写入 ``LOSS_REGISTRY``。
- ``policy_loss(loss_type="clipped_pg", task_type=..., ...)`` 会完成具身任务输入预处理、调用你的 Loss 函数并整理日志指标。

在 YAML 或命令行中启用：

.. code-block:: yaml

   algorithm:
     loss_type: clipped_pg
     clip_ratio_low: 0.2
     clip_ratio_high: 0.2

3. 打包与启动
-------------

按代码存放位置选择仓库内集成或外部仓库加载：

.. tab-set::

   .. tab-item:: 主仓库内注册（In-Tree）
      :sync: ext-in-tree

      将实现放入 ``rlinf/algorithms/advantages.py`` 或 ``rlinf/algorithms/losses.py``\ （所有 worker 导入时自动生效），然后用注册名称启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp \
           algorithm.adv_type=centered_reward algorithm.loss_type=clipped_pg

   .. tab-item:: 外部仓库注册（Out-of-Tree）
      :sync: ext-out-of-tree

      在外部扩展包的 ``register()`` 中导入对应模块，启动前导出 ``RLINF_EXT_MODULE``，确保每个 Ray worker 初始化时自动执行注册装饰器：

      .. code-block:: python

         # my_pkg/rlinf_ext.py
         def register():
             import my_pkg.my_advantages  # 执行 @register_advantage
             import my_pkg.my_losses      # 执行 @register_policy_loss

      .. code-block:: bash

         export RLINF_EXT_MODULE=my_pkg.rlinf_ext
         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp \
           algorithm.adv_type=centered_reward algorithm.loss_type=clipped_pg
