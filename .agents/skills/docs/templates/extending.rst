.. Extending guide template: copy this file, replace every <...>, and delete
.. these comments. Rules: .agents/skills/docs/rules/extending.md.

Add a Custom <Component>
========================

<One paragraph: what you build on this page, which RLinf registry or base class
it plugs into, and what runs once it is wired up.>

.. list-table:: At a Glance
   :header-rows: 1
   :widths: 26 74

   * - Item
     - Value
   * - Extension point
     - ``<registry decorator or enum>`` in ``<module path>``
   * - Base class / signature
     - ``<BaseClass>`` (implement ``<method_a>``, ``<method_b>``)
   * - Config selector
     - ``<yaml.key>: <your_name>``
   * - Out-of-tree loading
     - Set ``RLINF_EXT_MODULE=<your_package.module>`` (see :doc:`index`)

Prerequisites
-------------

- Working RLinf virtual environment (:doc:`../start/installation`).
- <Any upstream package or model/simulator asset.>

1. Implement the <Component>
----------------------------

<Explain the contract in plain words first, then show a minimal, runnable
implementation with real tensor shapes or outputs.>

.. code-block:: python

   <minimal runnable implementation>

2. Register and Select in Config
--------------------------------

Choose whether to register inside RLinf or from an external repository:

.. tab-set::

   .. tab-item:: In-Tree (Inside RLinf)
      :sync: ext-in-tree

      <In-tree registration call>

   .. tab-item:: Out-of-Tree (External Repo)
      :sync: ext-out-of-tree

      .. code-block:: python

         # my_pkg/rlinf_ext.py
         def register():
             <registration call>

      .. code-block:: bash

         export RLINF_EXT_MODULE=my_pkg.rlinf_ext

Select the registered name in your Hydra YAML:

.. code-block:: yaml

   <only the config keys the contributor sets>

3. Verify Locally on CPU
------------------------

Run a standalone Python check on CPU to confirm registration and the runtime
contract before launching a full distributed run:

.. code-block:: bash

   python -c '
   <standalone runnable CPU verification snippet>
   '

Next Steps
----------

- :doc:`index` — out-of-tree registration with ``RLINF_EXT_MODULE`` and other extension points.
- :doc:`../concepts/overview` — how RLinf schedules and runs components.
