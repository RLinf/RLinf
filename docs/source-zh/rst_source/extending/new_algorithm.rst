算法与规则奖励
==============

无需修改 RLinf 的 runner 或 worker，即可注册自定义优势估计函数、策略损失函数或基于规则的文本奖励。注册完成后，任意现有 YAML 配置都能通过 ``algorithm.adv_type``、``algorithm.loss_type`` 或 ``reward.reward_type`` 直接选用。

扩展点速览
----------

.. list-table::
   :header-rows: 1
   :widths: 22 28 24 26

   * - 扩展组件
     - 注册方式
     - 配置字段
     - 参考源码
   * - 优势估计（Advantage）
     - ``@register_advantage(name)``
     - ``algorithm.adv_type``
     - ``rlinf/algorithms/advantages.py``
   * - 策略损失（Policy Loss）
     - ``@register_policy_loss(name)``
     - ``algorithm.loss_type``
     - ``rlinf/algorithms/losses.py``
   * - 规则奖励（Rule Reward）
     - ``register_reward(name, cls)``
     - ``reward.reward_type``
     - ``rlinf/algorithms/rewards/__init__.py``

三类注册表均按进程独立维护：主仓内新增组件在模块导入时自动注册；外部仓库通过 ``RLINF_EXT_MODULE`` 在每个 worker 中完成注册（见 :ref:`out-of-tree-extensions`\ ）。

1. 自定义优势估计（Advantage）
------------------------------

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

2. 自定义策略损失（Policy Loss）
--------------------------------

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

- ``@register_policy_loss("clipped_pg")`` 将损失函数写入 ``LOSS_REGISTRY``。
- ``policy_loss(loss_type="clipped_pg", task_type=..., ...)`` 会完成具身任务输入预处理、调用你的损失函数并整理日志指标。

在 YAML 或命令行中启用：

.. code-block:: yaml

   algorithm:
     loss_type: clipped_pg
     clip_ratio_low: 0.2
     clip_ratio_high: 0.2

3. 自定义规则奖励（Rule-Based Reward）
--------------------------------------

在推理与智能体任务中，将 ``reward.use_reward_model`` 设为 ``False`` 时，``RewardWorker`` 会通过已注册的规则奖励类为文本生成结果打分。在 ``rlinf/algorithms/rewards/<name>/__init__.py``\ （或外部包）中实现 ``__init__(self, config)`` 与 ``get_reward(self, completions, answers, **kwargs)``，并通过 ``register_reward`` 注册。

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

在推理或智能体 YAML 中启用：

.. code-block:: yaml

   reward:
     use_reward_model: False
     reward_type: exact_match
     reward_scale: 1.0

4. 外部仓库注册
---------------

如需在 RLinf 仓库之外维护自定义算法，在扩展模块的 ``register()`` 中导入这些模块并导出 ``RLINF_EXT_MODULE``：

.. code-block:: python

   # my_pkg/rlinf_ext.py
   def register():
       import my_pkg.my_advantages  # 执行 @register_advantage
       import my_pkg.my_losses      # 执行 @register_policy_loss
       import my_pkg.my_rewards     # 执行 register_reward(...)

.. code-block:: bash

   export RLINF_EXT_MODULE=my_pkg.rlinf_ext
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp \
     algorithm.adv_type=centered_reward algorithm.loss_type=clipped_pg
