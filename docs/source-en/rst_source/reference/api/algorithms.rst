Algorithm Interface
===================

Look up the registry decorators and dispatch entrypoints in ``rlinf.algorithms.registry``. For mathematical objectives and configs, see :doc:`../algorithms/index`; to add a new advantage or loss function, see :doc:`../../extending/new_algorithm`.

Advantage Registry
------------------

.. autofunction:: rlinf.algorithms.registry.register_advantage

.. autofunction:: rlinf.algorithms.registry.get_adv_and_returns

.. autofunction:: rlinf.algorithms.registry.calculate_adv_and_returns

Policy Loss Registry
--------------------

.. autofunction:: rlinf.algorithms.registry.register_policy_loss

.. autofunction:: rlinf.algorithms.registry.get_policy_loss

.. autofunction:: rlinf.algorithms.registry.policy_loss

Loss Scale and Tool-Call Parser Registries
------------------------------------------

.. autofunction:: rlinf.algorithms.registry.register_loss_scale

.. autofunction:: rlinf.algorithms.registry.get_loss_scales

.. autofunction:: rlinf.algorithms.registry.register_toolcall_parser

.. autofunction:: rlinf.algorithms.registry.get_toolcall_parser
