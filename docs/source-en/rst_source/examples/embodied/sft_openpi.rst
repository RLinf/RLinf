Supervised Fine-Tuning
=======================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This page explains how to run **full-parameter supervised fine-tuning (SFT)** and **LoRA fine-tuning** with the RLinf framework. SFT is typically the first stage before reinforcement learning: the model imitates high-quality examples so RL can continue optimization with a strong prior.

Contents
----------

- How to configure full-parameter SFT and LoRA SFT in RLinf
- How to launch training on a single machine or multi-node cluster
- How to monitor and evaluate results


Supported datasets
--------------------

RLinf currently supports datasets in the LeRobot format, selected via **config_type**.

Supported formats include:

- pi0_maniskill
- pi0_libero
- pi0_aloha_robotwin
- pi0_franka_dagger
- pi05_libero
- pi05_maniskill
- pi05_metaworld
- pi05_calvin

You can also train with a custom dataset format. Refer to the files below:

1. In ``examples/sft/config/custom_sft_openpi.yaml``, set the data format.

.. code:: yaml

  model:
    openpi:
      config_name: "pi0_custom"

2. In ``rlinf/models/embodiment/openpi/__init__.py``, set the data format to ``pi0_custom``.

.. code:: python

    TrainConfig(
        name="pi0_custom",
        model=pi0_config.Pi0Config(),
        data=CustomDataConfig(
            repo_id="physical-intelligence/custom_dataset",
            base_config=DataConfig(
                prompt_from_task=True
            ),  # we need language instruction
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
            extra_delta_transform=True,  # True for delta action, False for abs_action
            action_train_with_rotation_6d=False,  # User can add extra config in custom dataset
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),

3. In ``rlinf/models/embodiment/openpi/dataconfig/custom_dataconfig.py``, define the custom dataset config.

.. code:: python

    class CustomDataConfig(DataConfig):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.repo_id = "physical-intelligence/custom_dataset"
            self.base_config = DataConfig(
                prompt_from_task=True
            )
            self.assets = AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets")
            self.extra_delta_transform = True
            self.action_train_with_rotation_6d = False

Normalization statistics for new LeRobot datasets
-------------------------------------------------

When you train OpenPI on a newly collected LeRobot dataset, compute dataset
normalization statistics before launching SFT. This is especially important for
a realworld collected dataset.

RLinf provides ``toolkits/replay_buffer/calculate_norm_stats.py`` to calculate norm_stats for ``state`` and ``actions``. You can use it like:

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/replay_buffer/calculate_norm_stats.py \
       --config-name pi0_franka_dagger \
       --repo-id franka_dagger

Notes:

- ``HF_LEROBOT_HOME`` must be set before running the script.
- ``config_name`` must match your custom openpi dataconfig used by training.
- ``repo_id`` must match your lerobot-format dataset name.

The script writes the generated stats under ``<assest_dir>/<exp_name>/<repo_id>/norm_stats.json``.

The OpenPI loader later reads the normalization stats from the ``<model_path>/<repo_id>`` at runtime.

Another practical tip for stable training is to manually check the normalization statistics for very small standard deviations or narrow q99–q01 ranges. Increasing the standard deviation or widening the q99–q01 gap can help stabilize training, especially in two-stage pipelines that transition from SFT to online training.


Training configuration
----------------------

Full examples live in:

- ``examples/sft/config/libero_sft_openpi.yaml``
- ``examples/sft/config/franka_dagger_sft_openpi.yaml``

A generic OpenPI SFT example looks like this:

.. code:: yaml

    cluster:
        num_nodes: 1                 # number of nodes
        component_placement:         # component → GPU mapping
            actor: 0-3

To enable LoRA fine-tuning, set ``actor.model.is_lora`` to True and configure ``actor.model.lora_rank``.

.. code:: yaml

    actor:
        model:
            is_lora: True
            lora_rank: 32

Dependency Installation
-----------------------

This section describes the dependency for the SFT of OpenPI model. 
For other models, please refer to the ``Dependency Installation`` section of the corresponding examples.

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    # For mainland China users, you can use the following for better download speed:
    # git clone https://ghfast.top/github.com/RLinf/RLinf.git
    git clone https://github.com/RLinf/RLinf.git
    cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

    docker run -it --rm --gpus all \
        --shm-size 20g \
        --network host \
        --name rlinf \
        -v .:/workspace/RLinf \
        rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
        # For mainland China users, you can use the following for better download speed:
        # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

    # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

    bash requirements/install.sh embodied --model openpi --env maniskill_libero
    source .venv/bin/activate

Launch scripts
----------------

First start the Ray cluster, then run the helper script:

.. code:: bash

   # return to repo root
   bash examples/sft/run_vla_sft.sh libero_sft_openpi

The same script works for generic text SFT; just swap the config file.

Real-World SFT Environment and Deployment
------------------------------------------

RLinf provides a **generic SFT environment** (``FrankaEnv-v1``) that lets you
define new real-world tasks entirely through YAML configuration, without writing
a custom environment class. This is useful for:

- Collecting SFT demonstration data on new tasks
- Deploying (evaluating) a trained policy on the real robot

Generic SFT Environment
~~~~~~~~~~~~~~~~~~~~~~~~

The env config template lives at
``examples/embodiment/config/env/realworld_franka_sft_env.yaml``.  Key fields
you should customise for your task:

.. code:: yaml

   override_cfg:
     task_description: "pick up the object and place it into the container"
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # goal pose [x,y,z,rx,ry,rz]
     reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]    # reset pose (above goal)
     max_num_steps: 300
     reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]  # success tolerance
     action_scale: [1.0, 1.0, 1.0]                         # [xyz, rpy, gripper]
     ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
     ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

Under the hood, ``FrankaEnv`` now accepts ``override_cfg`` as a plain dict and
uses a class-variable ``CONFIG_CLS`` to instantiate the dataclass config
(defaults to ``FrankaRobotConfig``).  Subclasses such as ``PegInsertionEnv`` and
``BottleEnv`` override ``CONFIG_CLS`` to their own dataclass while sharing the
same constructor.

Real-World Evaluation / Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A full evaluation config is provided at
``examples/embodiment/config/realworld_eval.yaml``.  It pairs the generic SFT
env with a Pi0 actor in **eval-only mode** (``runner.only_eval: True``).

Before running, replace the placeholders:

- ``ROBOT_IP`` — your Franka robot's IP address.
- ``MODEL_PATH`` — path to your trained checkpoint.

Then launch:

.. code:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_eval