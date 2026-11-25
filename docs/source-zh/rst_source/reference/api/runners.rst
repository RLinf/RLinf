Runner 接口
===========

查阅各任务族的顶层训练循环 runner 接口，用于编排 actor、rollout、环境、奖励与智能体 worker 组。

EmbodiedRunner
--------------

.. autoclass:: rlinf.runners.embodied_runner.EmbodiedRunner
   :members:
   :member-order: bysource

AsyncEmbodiedRunner
-------------------

.. autoclass:: rlinf.runners.async_embodied_runner.AsyncEmbodiedRunner
   :show-inheritance:
   :members:
   :member-order: bysource

AsyncPPOEmbodiedRunner
----------------------

.. autoclass:: rlinf.runners.async_ppo_embodied_runner.AsyncPPOEmbodiedRunner
   :show-inheritance:
   :members:
   :member-order: bysource

ReasoningRunner
---------------

.. autoclass:: rlinf.runners.reasoning_runner.ReasoningRunner
   :members:
   :member-order: bysource

AgentRunner
-----------

.. autoclass:: rlinf.runners.agent_runner.AgentRunner
   :show-inheritance:
   :members:
   :member-order: bysource

SFTRunner
---------

.. autoclass:: rlinf.runners.sft_runner.SFTRunner
   :members:
   :member-order: bysource
