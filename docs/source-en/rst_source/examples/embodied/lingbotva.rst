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
* **Observation**: head RGB image, left/right wrist RGB images, and proprioceptive state.
* **Action Space**: 16D end-effector actions.
* **Planner backend**: ``curobo``.
* **Execution type**: ``action_type: ee``.
* **Current support**: evaluation through RLinf rollout workers. An SFT entry is also included.

The validated configs keep:

* ``runner.only_eval=True``
* ``total_num_envs`` can be configured per task and available hardware.
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

   bash requirements/install.sh embodied --venv .venv-lingbotva --model lingbotva --env robotwin
   source .venv-lingbotva/bin/activate

Use a dedicated venv for LingBot-VA installation in RLinf. The install script clones the official
LingBot-VA repository into ``.venv-lingbotva/lingbot-va`` and writes
``LINGBOT_VA_REPO_PATH`` into ``.venv-lingbotva/bin/activate`` automatically.

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

This path should point to the **checkpoint root directory** downloaded by
``download_assets.sh``. In particular, RLinf expects:

.. code-block:: bash

   ${LINGBOT_VA_MODEL_PATH}/transformer/config.json

to exist.

5. Check ``attn_mode`` before evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before evaluation, open ``<model>/transformer/config.json`` and make sure ``attn_mode`` is set to ``"torch"`` or ``"flashattn"``.

Configuration Files
-------------------

The current RoboTwin evaluation configs are:

* ``examples/embodiment/config/robotwin_click_bell_lingbotva_eval.yaml``
* ``examples/embodiment/config/robotwin_place_empty_cup_lingbotva_eval.yaml``

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
   export LINGBOT_VA_REPO_PATH=/path/to/RLinf/.venv-lingbotva/lingbot-va
   export ROBOT_PLATFORM=ALOHA

Required meanings:

* ``ROBOTWIN_PATH``: the local RoboTwin repository root.
* ``LINGBOT_VA_MODEL_PATH``: the LingBot-VA checkpoint root directory downloaded by
  ``requirements/embodied/download_assets.sh``.
* ``LINGBOT_VA_REPO_PATH``: the official LingBot-VA repository root used by RLinf.
  If you install with
  ``bash requirements/install.sh embodied --venv .venv-lingbotva --model lingbotva --env robotwin``,
  this is typically ``<your_RLinf_repo>/.venv-lingbotva/lingbot-va``.

Optional override:

.. code-block:: bash

   # Only needed when calling eval_embodied_agent.py directly.
   export REPO_PATH=/path/to/RLinf

Launch Command
--------------

Recommended command:

.. code-block:: bash

   bash examples/embodiment/eval_embodiment.sh robotwin_click_bell_lingbotva_eval

If you call ``eval_embodied_agent.py`` directly, set ``REPO_PATH`` first and then run:

.. code-block:: bash

   python examples/embodiment/eval_embodied_agent.py \
     --config-path examples/embodiment/config \
     --config-name robotwin_click_bell_lingbotva_eval

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
