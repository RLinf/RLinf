RL with PolaRiS Simulation Platform
===================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/polaris.png
   :align: center
   :width: 90%

   PolaRiS (image: `PolaRiS <https://github.com/arhanjain/polaris>`__).

`PolaRiS <https://github.com/arhanjain/polaris>`__ is an Isaac Sim robotics
benchmark with Gaussian Splatting rendering for desktop manipulation. You'll use
RLinf to PPO-fine-tune OpenPI π₀ or π₀.₅ policies on DROID-style PolaRiS tasks.

Overview
--------

Fine-tune an OpenPI policy on PolaRiS with two RGB views, proprioception, and
chunked 8-dim actions.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      6 DROID desktop tasks

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1 GPU

| **You'll do:** install → download Isaac Sim + datasets + model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Isaac Sim · PolaRiS-Hub · an OpenPI checkpoint.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Task
     - Description
     - Env Config
   * - ``DROID-TapeIntoContainer``
     - Put the tape into the container.
     - ``polaris_droid_tapeintocontainer.yaml``
   * - ``DROID-PanClean``
     - Use the yellow sponge to scrub the blue-handle frying pan.
     - ``polaris_droid_panclean.yaml``
   * - ``DROID-BlockStackKitchen``
     - Place and stack blocks on the green tray.
     - ``polaris_droid_blockstackkitchen.yaml``
   * - ``DROID-FoodBussing``
     - Put all foods in the bowl.
     - ``polaris_droid_foodbussing.yaml``
   * - ``DROID-MoveLatteCup``
     - Put the latte art cup on top of the cutting board.
     - ``polaris_droid_movelattecup.yaml``
   * - ``DROID-OrganizeTools``
     - Put the scissors into the large container.
     - ``polaris_droid_organizetools.yaml``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - External RGB camera and wrist RGB camera at 224×224 plus 8-dim robot state.
   * - Action
     - 8-dim continuous action: 7 joint velocities plus gripper position.
   * - Reward
     - Task-completion reward from the PolaRiS environment.
   * - Prompt
     - The task description in ``init_params.task_description``.

Installation
------------

.. include:: _setup_common.rst

**Docker image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-polaris

   # For mainland China users:
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-polaris

Switch to the OpenPI virtual environment inside the image:

.. code:: bash

   source switch_env openpi

**Custom environment**

Install PolaRiS with OpenPI dependencies:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

Download Isaac Sim
~~~~~~~~~~~~~~~~~~

Download Isaac Sim 5.1.0 and initialize its shell environment:

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
   source ./setup_conda_env.sh

.. warning::

   Run ``source ./setup_conda_env.sh`` in every new terminal before launching PolaRiS.

Download the Datasets
---------------------

Download the evaluation scenes and initial conditions:

.. code:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub
   export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

Optionally download co-training demonstrations:

.. code:: bash

   hf download owhan/PolaRiS-datasets --repo-type=dataset --local-dir ./PolaRiS-datasets

Download the Model
------------------

Download the checkpoint for the OpenPI model you plan to fine-tune.

**OpenPI π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-Polaris-droid_jointpos

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir RLinf-Pi05-Polaris-droid_jointpos

**OpenPI π₀**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Polaris-droid_jointpos

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir RLinf-Pi0-Polaris-droid_jointpos

.. include:: _model_path.rst

Run It
------

Pick one training config and launch from a terminal where Isaac Sim is initialized:

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - Recipe
     - Config
     - Command suffix
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/polaris_tapeintocontainer_ppo_openpi_pi05.yaml``
     - ``polaris_tapeintocontainer_ppo_openpi_pi05``
   * - π₀ + PPO
     - ``examples/embodiment/config/polaris_tapeintocontainer_ppo_openpi.yaml``
     - ``polaris_tapeintocontainer_ppo_openpi``

.. code:: bash

   source /path/to/isaac_sim/setup_conda_env.sh
   export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

   bash examples/embodiment/run_embodiment.sh polaris_tapeintocontainer_ppo_openpi_pi05
   bash examples/embodiment/run_embodiment.sh polaris_tapeintocontainer_ppo_openpi

What this does:

1. Starts the embodied training entrypoint with the selected Hydra config.
2. Creates Ray workers for the actor, rollout, and PolaRiS env components.
3. Runs PPO with chunked OpenPI actions and Gaussian Splatting-rendered observations.

Run standalone evaluation through the :doc:`PolaRiS evaluation guide <../../evaluations/guides/polaris>`.
It owns ``POLARIS_DATA_PATH``, the available eval configs
(``polaris_tapeintocontainer_openpi_pi05_eval`` and ``polaris_movelattecup_openpi_eval``),
and result interpretation.

.. note::

   Training configs default to ``polaris_droid_tapeintocontainer``. To switch tasks,
   change the Hydra env defaults to another ``polaris_droid_*`` env config and keep
   ``POLARIS_DATA_PATH`` pointed at ``PolaRiS-Hub``.

A few PolaRiS-specific fields are worth knowing when tuning the action/rendering pipeline:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``open_loop_horizon``
     - Frequency of **high-quality** Gaussian-Splatting rendering. Within an action chunk,
       high-quality rendering runs every ``open_loop_horizon`` steps while intermediate steps
       use low-quality rendering to speed up the simulation.
   * - ``num_action_chunks``
     - Number of action steps the model generates at a time (e.g. ``15``).
   * - ``num_images_in_input``
     - Number of camera images fed to the policy (e.g. ``2``: external + wrist camera).
   * - ``config_name``
     - OpenPI config / data format selector (e.g. ``pi05_droid_polaris`` for the DROID data format).

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable evaluation videos in the env config when needed:

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval
