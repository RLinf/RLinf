Envs & Logging
==============

Use these guides to configure heterogeneous training environments and experiment logging before launching a run.

.. note::

   For full YAML field schemas and metric namespaces, see :doc:`Config & Metrics <../../reference/training/index>`. For adapter settings on VLA and diffusion models, see :doc:`LoRA Fine-Tuning <../lora>`.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`Logging & Visualization <../logger>`
     - TensorBoard, W&B, and SwanLab setup, including offline logging and multi-node API keys.
   * - :doc:`Heterogeneous Envs <../hetero_envs>`
     - Run different env configurations, such as ManiSkill and LIBERO, for one policy in one run.

.. toctree::
   :hidden:

   Logging & Visualization <../logger>
   Heterogeneous Envs <../hetero_envs>
