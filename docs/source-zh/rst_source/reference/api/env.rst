Env 接口
========

查阅分布式环境 worker、通过 ``env_type`` 解析环境类的工厂接口，以及真机环境封装类。接入新仿真器的步骤见 :doc:`../../extending/new_env`。

EnvWorker
---------

.. autoclass:: rlinf.workers.env.env_worker.EnvWorker
   :show-inheritance:
   :members:
   :member-order: bysource

环境工厂
--------

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
