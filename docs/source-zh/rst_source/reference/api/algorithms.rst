Algorithm 接口
==============

查阅 ``rlinf.algorithms.registry`` 中的注册装饰器与统一调度入口。算法目标函数与配置见 :doc:`../algorithms/index`，新增 Advantage 或 Loss 函数的步骤见 :doc:`../../extending/new_algorithm`。

Advantage 注册表
----------------

.. autofunction:: rlinf.algorithms.registry.register_advantage

.. autofunction:: rlinf.algorithms.registry.get_adv_and_returns

.. autofunction:: rlinf.algorithms.registry.calculate_adv_and_returns

Policy Loss 注册表
------------------

.. autofunction:: rlinf.algorithms.registry.register_policy_loss

.. autofunction:: rlinf.algorithms.registry.get_policy_loss

.. autofunction:: rlinf.algorithms.registry.policy_loss

Loss Scale 与工具调用解析器注册表
---------------------------------

.. autofunction:: rlinf.algorithms.registry.register_loss_scale

.. autofunction:: rlinf.algorithms.registry.get_loss_scales

.. autofunction:: rlinf.algorithms.registry.register_toolcall_parser

.. autofunction:: rlinf.algorithms.registry.get_toolcall_parser
