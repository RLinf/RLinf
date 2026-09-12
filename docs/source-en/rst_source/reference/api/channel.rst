Channel API
===========

Use these interfaces to create queues and customize their enqueue and consumer
routing behavior. Start with :doc:`../../concepts/channel` for the execution
model and a minimal registered-hook example.

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
