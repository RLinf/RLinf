:orphan:

LingBot-VA on RoboTwin
======================

This document describes how to evaluate the official LingBot-VA RoboTwin checkpoint in RLinf.

The current integration focuses on RoboTwin evaluation and keeps the official LingBot-VA inference workflow, including 16D end-effector actions, ``add_init_pose``, key-frame updates, and ``compute_kv_cache`` between chunks.

Overview
--------

LingBot-VA is currently supported in RLinf for **RoboTwin evaluation only**.

The current integration has the following characteristics:

* **Environment**: RoboTwin 2.0 tasks executed through RLinf ``RoboTwinEnv``.
* **Observation**: head RGB image, left/right wrist RGB images, and 14D proprioception.
* **Action Space**: 16D end-effector actions.
* **Planner backend**: ``curobo``.
* **Execution type**: ``action_type: ee``.
* **Current support**: evaluation only, one LingBot-VA websocket session per rollout worker.

The validated configs keep:

* ``runner.only_eval=True``
* ``total_num_envs: 1``
* ``enable_offload: false``

Dependency Installation
-----------------------

1. Clone the RLinf repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install LingBot-VA and RoboTwin support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash requirements/install.sh embodied --model lingbotva --env robotwin
   source .venv/bin/activate

The install script clones the official LingBot-VA repository into ``.venv/lingbot-va`` and writes ``LINGBOT_VA_REPO_PATH`` into ``.venv/bin/activate`` automatically.

3. Prepare RoboTwin assets
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the RoboTwin ``RLinf_support`` branch and download its assets:

.. code-block:: bash

   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
   cd RoboTwin
   bash script/_download_assets.sh

4. Download the LingBot-VA checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash requirements/embodied/download_assets.sh --dir /path/to/assets --assets lingbotva

Set ``LINGBOT_VA_MODEL_PATH`` to:

.. code-block:: bash

   /path/to/assets/.cache/lingbotva/lingbot-va-posttrain-robotwin

5. Check ``attn_mode`` before evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before evaluation, open ``<model>/transformer/config.json`` and make sure ``attn_mode`` is set to ``"torch"`` or ``"flashattn"``.

Configuration Files
-------------------

The current RoboTwin evaluation configs are:

* ``examples/embodiment/config/robotwin_click_bell_eval_lingbotva.yaml``
* ``examples/embodiment/config/robotwin_place_empty_cup_eval_lingbotva.yaml``

These configs keep the validated setup:

* ``planner_backend: curobo``
* ``action_type: ee``
* ``action_dim: 16``
* ``num_action_chunks: 32``
* ``ignore_terminations: false``
* ``enable_offload: false``

Environment Variables
---------------------

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   export LINGBOT_VA_MODEL_PATH=/path/to/assets/.cache/lingbotva/lingbot-va-posttrain-robotwin
   export ROBOT_PLATFORM=ALOHA

Optional overrides:

.. code-block:: bash

   # Only needed when calling eval_embodied_agent.py directly.
   export REPO_PATH=/path/to/RLinf

   # Only needed when overriding the LingBot-VA repo cloned by install.sh.
   export LINGBOT_VA_REPO_PATH=/path/to/lingbot-va

Launch Command
--------------

Recommended command:

.. code-block:: bash

   bash examples/embodiment/eval_embodiment.sh robotwin_click_bell_eval_lingbotva

If you call ``eval_embodied_agent.py`` directly, set ``REPO_PATH`` first and then run:

.. code-block:: bash

   python examples/embodiment/eval_embodied_agent.py \
     --config-path examples/embodiment/config \
     --config-name robotwin_click_bell_eval_lingbotva

Evaluation Results and Visualization
------------------------------------

The current validated tasks are:

* ``click_bell``
* ``place_empty_cup``

If TensorBoard logging is enabled, you can inspect results with:

.. code-block:: bash

   tensorboard --logdir ../results --port 6006

If evaluation video logging is enabled, videos are saved under:

.. code-block:: bash

   <runner.logger.log_path>/video/eval

Notes
-----

* ``center_crop`` should be set explicitly per task.
* ``max_episode_steps`` and ``max_steps_per_rollout_epoch`` should stay aligned with RLinf chunk rollout scheduling.
* The current integration keeps the official LingBot-VA chunk/key-frame/KV-cache evaluation rhythm.
