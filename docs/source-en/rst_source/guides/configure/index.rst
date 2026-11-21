Configure
=========

Use these guides when you need to edit YAML, logging, or workload-specific
settings before launching a run.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`Basic Configuration <../basic_config>`
     - How RLinf config files are structured and overridden.
   * - :doc:`Embodied Configuration <../embodiment_config>`
     - Config keys specific to embodied training.
   * - :doc:`Agentic Configuration <../agentic_config>`
     - Config keys specific to agentic and reasoning training.
   * - :doc:`Logging <../logger>`
     - TensorBoard, wandb, and swanlab logger setup.
   * - :doc:`Interpreting success_once Fluctuations <../eval_success_once_noise>`
     - Tell statistical noise in embodied eval metrics apart from real instability.

.. toctree::
   :hidden:

   Basic Configuration <../basic_config>
   Embodied Configuration <../embodiment_config>
   Agentic Configuration <../agentic_config>
   Logging <../logger>
   Interpreting success_once Fluctuations <../eval_success_once_noise>
