Env Interface
=============

Use these environment APIs when launching simulator or real-world environment workers, resolving an environment class from ``env_type``, or implementing custom embodied tasks. To integrate a new simulator, see :doc:`../../extending/new_env`.

EnvWorker
---------

.. autoclass:: rlinf.workers.env.env_worker.EnvWorker
   :show-inheritance:
   :members:
   :member-order: bysource

Environment Factory
-------------------

.. autoclass:: rlinf.envs.SupportedEnvType
   :members:
   :undoc-members:
   :member-order: bysource

.. autofunction:: rlinf.envs.get_env_cls

RealWorldEnv
------------

.. autoclass:: rlinf.envs.real.env.RealWorldEnv
   :show-inheritance:
   :members:
   :member-order: bysource
