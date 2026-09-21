Channel API
===========

使用这些接口创建队列，并定制入队转换和消费者路由行为。执行模型和最小注册 hook
案例参见 :doc:`../../concepts/channel`。

Channel
-------

.. autoclass:: rlinf.scheduler.Channel
   :members:
   :member-order: bysource
   :no-index:
   :class-doc-from: class

Hook Context
------------

.. autoclass:: rlinf.scheduler.ChannelContext
   :members:
   :member-order: bysource

Collector
---------

.. autoclass:: rlinf.scheduler.Collector
   :members:
   :member-order: bysource

.. autofunction:: rlinf.scheduler.register_collector

Dispatcher
----------

.. autoclass:: rlinf.scheduler.Dispatcher
   :members:
   :member-order: bysource

.. autofunction:: rlinf.scheduler.register_dispatcher
