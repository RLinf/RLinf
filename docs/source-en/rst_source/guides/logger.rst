Logging & Visualization
=======================

Track training loss, rewards, evaluation scores, and step timings in real time with TensorBoard, Weights & Biases (W&B), or SwanLab. For the full list of logged metric namespaces (``train/``, ``rollout/``, ``env/``, ``eval/``, and ``time/``), see :doc:`Training Metrics <../reference/metrics>`.

Supported Backends
------------------

- `TensorBoard <https://www.tensorflow.org/tensorboard>`__: Local event files for scalar plots, histograms, and run comparison.
- `Weights & Biases (W&B) <https://wandb.ai/site/>`__: Cloud experiment tracking, artifact versioning, and team dashboards, with an offline sync workflow for air-gapped clusters.
- `SwanLab <https://pypi.org/project/swanlab/>`__: Lightweight cloud or local experiment tracking with metric plots, hyperparameters, and hardware stats.

Enable Logger Backends
----------------------

List the backends you want under ``runner.logger.logger_backends`` in your YAML config:

.. code-block:: yaml

   runner:
     experiment_name: grpo-1.5b
     output_dir: ./logs
     logger:
       log_path: ${runner.output_dir}/${runner.experiment_name}
       project_name: rlinf
       experiment_name: ${runner.experiment_name}
       logger_backends: ["tensorboard", "wandb", "swanlab"]  # choose any subset

RLinf also accepts optional backend settings under ``runner.logger``: ``wandb_entity`` (W&B team or username), ``wandb_proxy`` (HTTPS proxy URL for W&B), and ``swanlab_mode`` (``"cloud"``, ``"local"``, ``"offline"``, or ``"disabled"``; defaults to ``"cloud"``).

Under ``runner.logger.log_path``, RLinf writes each backend to its own subdirectory:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Subdirectory
     - Contents
   * - ``tensorboard/``
     - TensorBoard event files and the resolved ``config.yaml``.
   * - ``wandb/``
     - W&B local run directory and offline sync files.
   * - ``swanlab/``
     - SwanLab run logs and event files.
   * - ``checkpoints/``
     - Saved model and optimizer checkpoints.

.. tip::

   All three backends can run in parallel. Pass an empty list ``[]`` to disable metric logging.

TensorBoard
-----------

Start TensorBoard against the run directory to inspect scalar curves locally:

.. code-block:: bash

   tensorboard --logdir ./logs/grpo-1.5b/tensorboard --port 6006

Open ``http://localhost:6006`` in your browser.

Remote Viewing
~~~~~~~~~~~~~~

On a remote GPU server, bind TensorBoard to all network interfaces so it accepts external or tunneled connections:

.. code-block:: bash

   tensorboard --logdir ./logs/grpo-1.5b/tensorboard --host 0.0.0.0 --port 6006

If port ``6006`` is blocked by a firewall, forward it from your local machine over SSH and open ``http://localhost:6006`` locally:

.. code-block:: bash

   ssh -L 6006:localhost:6006 <user>@<remote_host>

Weights & Biases (W&B)
----------------------

Online Setup
~~~~~~~~~~~~

1. Create an account at `wandb.ai <https://wandb.ai>`__ and copy your API key.
2. Authenticate once on the machine that launches training:

   .. code-block:: bash

      wandb login

RLinf starts a new W&B run automatically under ``runner.logger.project_name`` and ``runner.logger.experiment_name``. Set ``runner.logger.wandb_entity`` when logging to a shared team, or ``runner.logger.wandb_proxy`` when W&B traffic goes through an HTTPS proxy.

Offline Mode
~~~~~~~~~~~~

On clusters without outbound internet access, run W&B in offline mode so metrics save locally under ``<log_path>/wandb/`` without blocking training:

.. code-block:: bash

   export WANDB_MODE=offline

After training finishes (or from a login node with internet access), upload the saved runs to W&B:

.. code-block:: bash

   wandb sync ./logs/grpo-1.5b/wandb/wandb/offline-run-*

Pass ``--sync-all`` to upload every unsynced run in a directory at once.

SwanLab
-------

1. Register at `swanlab.ai <https://swanlab.ai>`__ and copy your access token.
2. Authenticate once on the launch machine:

   .. code-block:: bash

      swanlab login

For local-only or offline logging without uploading to SwanLab Cloud, set ``runner.logger.swanlab_mode`` to ``"local"`` or ``"offline"`` in your YAML config.

Pass API Keys in Multi-Node Ray Clusters
----------------------------------------

RLinf initializes the metric logger on the head node driver process. In multi-node clusters where worker processes or job launchers do not share interactive shell logins, pass credentials through environment variables instead of running interactive ``wandb login`` or ``swanlab login``.

Export Before Starting Ray
~~~~~~~~~~~~~~~~~~~~~~~~~~

Export the API keys on the head node before starting Ray and the training script (see :doc:`Multi-Node Ray Cluster Setup <multi_node>`). Ray captures the shell environment at ``ray start`` time, so any worker started on that node also inherits the variables:

.. code-block:: bash

   export WANDB_API_KEY=<your_wandb_api_key>
   export SWANLAB_API_KEY=<your_swanlab_api_key>

Configure Per-Job Environment Variables in YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When jobs run in a shared Ray cluster and you prefer per-job environment injection, declare the keys under ``cluster.node_groups[].env_configs[].env_vars`` in your config:

.. code-block:: yaml

   cluster:
     node_groups:
       - label: "default"
         node_ranks: all
         env_configs:
           - node_ranks: all
             env_vars:
               - WANDB_API_KEY: "${oc.env:WANDB_API_KEY}"
               - SWANLAB_API_KEY: "${oc.env:SWANLAB_API_KEY}"

.. warning::

   Avoid committing plaintext API keys into YAML files. Reference shell variables with ``${oc.env:WANDB_API_KEY}`` or export them in your job submission environment.

See Also
--------

- :doc:`Training Metrics <../reference/metrics>`: definitions and debugging ranges for ``train/``, ``rollout/``, ``env/``, ``eval/``, and ``time/``.
- :doc:`Basic Configuration <basic_config>`: full ``runner`` and ``cluster`` YAML fields.
