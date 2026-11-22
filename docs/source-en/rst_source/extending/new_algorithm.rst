Algorithms & Rule-Based Rewards
===============================

Register a custom advantage estimator, policy loss, or rule-based text reward without touching RLinf's runners or workers. After registration, any existing YAML config can select your component through ``algorithm.adv_type``, ``algorithm.loss_type``, or ``reward.reward_type``.

Extension Points at a Glance
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 24 26

   * - Component
     - Register With
     - Selected By
     - Reference File
   * - Advantage estimator
     - ``@register_advantage(name)``
     - ``algorithm.adv_type``
     - ``rlinf/algorithms/advantages.py``
   * - Policy loss
     - ``@register_policy_loss(name)``
     - ``algorithm.loss_type``
     - ``rlinf/algorithms/losses.py``
   * - Rule-based reward
     - ``register_reward(name, cls)``
     - ``reward.reward_type``
     - ``rlinf/algorithms/rewards/__init__.py``

All three registries are per-process. In-tree additions register at module import; out-of-tree repositories register through ``RLINF_EXT_MODULE`` (see :ref:`out-of-tree-extensions`).

1. Custom Advantage Estimator
-----------------------------

Implement a function decorated with ``@register_advantage(name)`` in ``rlinf/algorithms/advantages.py`` (or in your own extension module). Accept ``**kwargs`` so RLinf can pass shared batch fields, and return a ``(advantages, returns)`` tuple (set ``returns`` to ``None`` for critic-free methods).

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

How the calls fit together:

- ``@register_advantage("centered_reward")`` stores the function under its lowercase name in ``ADV_REGISTRY``.
- ``calculate_adv_and_returns(adv_type="centered_reward", task_type=..., ...)`` preprocesses batch tensors, calls your function, and formats the output for the actor worker. Embodied tasks dispatch directly to any registered ``adv_type``; for reasoning tasks, also add a matching branch in ``preprocess_reasoning_advantages_inputs()`` in ``rlinf/algorithms/utils.py``.

Select it in YAML or on the command line:

.. code-block:: yaml

   algorithm:
     adv_type: centered_reward
     normalize_advantages: True

Verify locally on CPU without launching Ray:

.. code-block:: bash

   python -c '
   import torch
   import rlinf.algorithms.advantages  # triggers in-tree registrations
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

2. Custom Policy Loss
---------------------

Implement a function decorated with ``@register_policy_loss(name)`` in ``rlinf/algorithms/losses.py`` (or in your extension module). Accept ``**kwargs``, assert ``float32`` inputs for numerical stability, and return ``(loss, metrics_dict)`` where ``loss`` is a scalar tensor and every metric value is a detached scalar tensor or float.

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

How the calls fit together:

- ``@register_policy_loss("clipped_pg")`` registers the loss in ``LOSS_REGISTRY``.
- ``policy_loss(loss_type="clipped_pg", task_type=..., ...)`` preprocesses embodied inputs, calls your function, and post-processes logged metrics.

Select it in YAML or via a CLI override:

.. code-block:: yaml

   algorithm:
     loss_type: clipped_pg
     clip_ratio_low: 0.2
     clip_ratio_high: 0.2

3. Custom Rule-Based Reward
---------------------------

For reasoning and agentic tasks, ``RewardWorker`` scores text completions with a registered Python class when ``reward.use_reward_model`` is ``False``. Create a module under ``rlinf/algorithms/rewards/<name>/__init__.py`` (or in your external package) implementing ``__init__(self, config)`` and ``get_reward(self, completions, answers, **kwargs)``, then register it with ``register_reward``.

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

How the calls fit together:

- ``__init__(self, config)`` receives the ``cfg.reward`` section from Hydra.
- ``get_reward(self, completions, answers, **kwargs)`` returns one float score per completion. Set ``reward.use_prompt: True`` in YAML if you also want ``kwargs["prompts"]`` passed in.
- ``register_reward("exact_match", ExactMatchReward)`` adds the class to ``reward_registry`` so ``get_rule_based_reward_class("exact_match")`` instantiates it inside ``RewardWorker``.

Select it in your reasoning or agentic YAML:

.. code-block:: yaml

   reward:
     use_reward_model: False
     reward_type: exact_match
     reward_scale: 1.0

4. Out-of-Tree Packaging and Testing
------------------------------------

To keep custom algorithms outside the RLinf tree, register them inside your extension module's ``register()`` hook and export ``RLINF_EXT_MODULE``:

.. code-block:: python

   # my_pkg/rlinf_ext.py
   def register():
       import my_pkg.my_advantages  # executes @register_advantage
       import my_pkg.my_losses      # executes @register_policy_loss
       import my_pkg.my_rewards     # executes register_reward(...)

.. code-block:: bash

   export RLINF_EXT_MODULE=my_pkg.rlinf_ext
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp \
     algorithm.adv_type=centered_reward algorithm.loss_type=clipped_pg

Add a CPU pytest unit test to ``tests/unit_tests/test_utils.py`` verifying registry dispatch, output shapes, mask handling, and gradient flow:

.. code-block:: python

   def test_centered_reward_and_clipped_pg():
       import torch

       from rlinf.algorithms.registry import (
           calculate_adv_and_returns,
           policy_loss,
       )

       out = calculate_adv_and_returns(
           adv_type="centered_reward",
           task_type="embodied",
           reward_type="chunk_level",
           logprob_type="chunk_level",
           rewards=torch.tensor([[[1.0], [0.0]], [[0.5], [0.0]]]),
           dones=torch.zeros(3, 2, 1, dtype=torch.bool),
           loss_mask=torch.ones(2, 2, 1, dtype=torch.bool),
           group_size=2,
           normalize_advantages=True,
       )
       advantages = out["advantages"]
       assert advantages.shape == (2, 2, 1)
       assert abs(advantages.mean().item()) < 1e-4

       logprobs = torch.zeros(2, 2, 1, dtype=torch.float32, requires_grad=True)
       old_logprobs = torch.zeros(2, 2, 1, dtype=torch.float32)
       loss_mask = torch.ones(2, 2, 1, dtype=torch.bool)
       loss, metrics = policy_loss(
           loss_type="clipped_pg",
           task_type="embodied",
           logprobs=logprobs,
           old_logprobs=old_logprobs,
           advantages=advantages.float(),
           loss_mask=loss_mask,
           clip_ratio_low=0.2,
           clip_ratio_high=0.2,
       )
       assert loss.ndim == 0 and torch.isfinite(loss)
       assert "actor/policy_loss" in metrics and "actor/ratio" in metrics
       loss.backward()
       assert logprobs.grad is not None

Run the unit tests with:

.. code-block:: bash

   export PYTHONPATH=$(pwd):$(pwd)/tests/unit_tests
   pytest tests/unit_tests/test_utils.py
