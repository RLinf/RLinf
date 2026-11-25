Rule-Based Reward
=================

Register a deterministic Python reward scorer for reasoning and agentic tasks without modifying RLinf's runners or workers. When ``reward.use_reward_model`` is ``False``, ``RewardWorker`` resolves ``reward.reward_type`` through ``reward_registry``.

Extension Point at a Glance
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 24 26

   * - Component
     - Register With
     - Selected By
     - Reference File
   * - Rule-based reward
     - ``register_reward(name, cls)``
     - ``reward.reward_type``
     - ``rlinf/algorithms/rewards/__init__.py``

1. Implement and Register the Scorer
------------------------------------

Create a module under ``rlinf/algorithms/rewards/<name>/__init__.py`` (or in your external package) implementing ``__init__(self, config)`` and ``get_reward(self, completions, answers, **kwargs)``, then register it with ``register_reward``:

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

2. Select in YAML and Verify Locally
------------------------------------

Enable your reward class in a reasoning or agentic YAML:

.. code-block:: yaml

   reward:
     use_reward_model: False
     reward_type: exact_match
     reward_scale: 1.0

Verify registration and scoring locally on CPU without launching Ray:

.. code-block:: bash

   python -c '
   from omegaconf import OmegaConf
   from rlinf.algorithms.rewards import get_rule_based_reward_class

   reward_cls = get_rule_based_reward_class("math")
   scorer = reward_cls(OmegaConf.create({"reward_scale": 1.0}))
   print("loaded:", type(scorer).__name__)
   '

3. Package and Launch
---------------------

.. tab-set::

   .. tab-item:: In-Tree (Inside RLinf)
      :sync: ext-in-tree

      Import your module in ``rlinf/algorithms/rewards/__init__.py`` so every worker registers it automatically, then override ``reward.reward_type`` at launch.

   .. tab-item:: Out-of-Tree (External Repo)
      :sync: ext-out-of-tree

      Import your reward module inside your package's ``register()`` hook and export ``RLINF_EXT_MODULE`` before launching (see :ref:`out-of-tree-extensions`):

      .. code-block:: python

         # my_pkg/rlinf_ext.py
         def register():
             import my_pkg.my_rewards  # executes register_reward(...)

      .. code-block:: bash

         export RLINF_EXT_MODULE=my_pkg.rlinf_ext
