OGPO Training in ManiSkill
==========================

**OGPO (Off-Policy Generative Policy Optimization)** is an off-policy
full-policy finetuning method for generative control policies. It uses Q-value
estimates to construct advantages for groups of sampled actions and optimizes
the generative policy with a PPO-style objective.

The example in this repository targets ManiSkill's ``PickCube-v1`` task.

For the algorithm and original implementation, see the
`paper <https://arxiv.org/abs/2605.03065>`_ and
`official GitHub repository <https://github.com/simchowitzlabpublic/OGPO_public>`_.

Overview
--------

The example runs synchronous training on a single GPU.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      OGPO

   .. grid-item-card:: Models
      :text-align: center

      State-based flow policy

   .. grid-item-card:: Environments / Data
      :text-align: center

      ManiSkill · PickCube

   .. grid-item-card:: Training
      :text-align: center

      Online

Provided Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Task
     - Configuration
   * - ``PickCube-v1``
     - ``examples/embodiment/config/maniskill_ogpo_flow_state.yaml``

Installation
------------

Follow :doc:`ManiSkill <maniskill>` and
:doc:`Installation <../../start/installation>` to prepare the environment.
Activate it and run the following commands from the RLinf repository root.

Run It
------

In the YAML file above, set ``runner.ckpt_path`` to the actual BC checkpoint path
compatible with the configured model:

.. code-block:: yaml

   runner:
     ckpt_path: /path/to/your/ogpo_bc.pt

Choose an available GPU through ``cluster.component_placement``, then launch:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ogpo_flow_state

The script writes logs under ``logs/<timestamp>-maniskill_ogpo_flow_state/``.

Visualization
-------------

View training and evaluation in TensorBoard:

.. code-block:: bash

   tensorboard --logdir ./logs

Evaluation metric: ``eval/success_once``.
See :doc:`Training metrics <../../reference/metrics>` for metric definitions.
