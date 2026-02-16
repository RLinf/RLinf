RL-based Sim-Real Co-Training
==============================

This example shows how to use the RLinf framework for **sim-real co-training**
(Sim-Real Co-Training) of the π\ :sub:`0.5`\  model. It provides a simulation
environment, corresponding real and simulation datasets, and the full workflow
to run co-training in that setting.

Co-training combines **PPO** in simulation with **SFT** on real data so the
policy improves task success in sim while retaining real-world priors and
avoiding sim-only overfitting that hurts sim-to-real transfer.

For technical details, see the paper: *RLinf-Co: Reinforcement Learning–Based Sim–Real Co-Training for VLA Models*.

After training, the model is expected to support:

1. **Visual understanding**: Process RGB images from the robot camera.
2. **Language understanding**: Interpret natural-language task descriptions.
3. **Action generation**: Produce precise robot actions (position, rotation, gripper).
4. **Co-evolution**: Improve via RL in simulation while staying grounded via SFT on real data.


Environment
-----------

**Note:** This example provides a single demo environment. In practice, you should collect data and build a matching sim scene for your own setup.

**Real-world setup**

- **Environment**: Franka Emika Panda arm, RealSense camera.
- **Task**: Pick and place — place an object from the table into a bowl.
- **Observation**: Third-person RGB (640×480).
- **Language**: Task description from the environment.
- **Action space**: 7D continuous (x, y, z, roll, pitch, yaw, gripper open/close).

**Simulation (digital twin)**

Built with ManiSkill3:

- **Digital twin**: Aligned with the real setup in layout, camera view, task logic, language, and action space.
- **Dynamics**: Tuned to approximate real-world physics.


Algorithm
---------

This example uses **RL-Co**, combining:

1. **PPO (Proximal Policy Optimization)**
   - GAE for advantage estimation
   - Ratio-based policy clipping
   - Value clipping and entropy regularization

2. **SFT (Supervised Fine-Tuning)**
   - Real-world trajectory data as supervision alongside RL to avoid sim-only overfitting and preserve sim-to-real transfer.


Dependency installation
------------------------

1. Clone the RLinf repo
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For faster downloads in mainland China you can use:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
   # For faster image pull in mainland China you can use:
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

Then switch to the OpenPI env inside the container:

.. code:: bash

   source switch_env openpi

**Option 2: Local install**

.. code:: bash

   # Add --use-mirror for faster install in some regions
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate


ManiSkill assets
~~~~~~~~~~~~~~~~~~

Refer to the :doc:`ManiSkill example <maniskill>` for base asset setup, then download the assets required for this example:

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill/assets
   # For faster download in some regions you can set:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/RLCo-maniskill-assets --include "custom_assets/*" --local-dir .


Stage I: SFT pretraining
------------------------

Stage I injects real and sim data via supervised learning before RL. You can either train yourself or use a provided checkpoint.

**Option A: SFT with real + sim data**

We provide a LeRobot-format dataset (50 real + 1499 sim trajectories) at `RLinf/RLCo-Example-Mix-Data <https://huggingface.co/datasets/RLinf/RLCo-Example-Mix-Data>`_.

1. **Download the dataset**:

.. code:: bash

   # For faster download in some regions you can set:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/RLCo-Example-Mix-Data --local-dir RLCo-Example-Mix-Data

2. Run SFT using `OpenPi <https://github.com/Physical-Intelligence/openpi>`_ or the :doc:`SFT example <sft>`.

**Option B: Use an SFT checkpoint**

Skip training and use the provided SFT checkpoint:

.. code:: bash

   # Download the Spatial-Object-Goal model (choose one method)
   # Method 1: git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT

   # Method 2: huggingface-hub
   # For faster download in some regions you can set:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT --local-dir RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT

Stage II: Sim-real co-training (RL)
-----------------------------------

This stage adds SFT loss into the PPO loop for joint optimization.

**Data**

Download the 50 real trajectories in LeRobot format used for co-training:

.. code:: bash

   # For faster download in some regions you can set:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/RLCo-Example-Real-Data --local-dir RLCo-Example-Real-Data

**Important config**

The config ``maniskill_ppo_co_training_openpi_pi05.yaml`` is provided. For general PPO settings see :doc:`π₀ and π₀.₅ RL training <pi0>`. Co-training-specific options:

**Model paths**

Point ``model_path`` to your SFT checkpoint and ``sft_data_path`` to the real data path:

.. code-block:: yaml

   rollout:
      model:
         model_path: /path/to/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT
   actor:
      sft_data_path: /path/to/RLCo-Example-Real-Data
      model:
         model_path: /path/to/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT

**Co-training options**

.. code-block:: yaml

   actor:
       model:
           openpi:
               config_name: "pi05_maniskill_sim_real_co_training"
       enable_sft_co_train: True
       sft_loss_weight: 0.2

- ``enable_sft_co_train``: Set to ``True`` to enable co-training; ``False`` for PPO-only.
- ``sft_loss_weight``: Weight :math:`\beta` for the SFT term (:math:`\mathcal{L}_{SFT}`) in the total loss.

The dataconfig ``pi05_maniskill_sim_real_co_training`` is defined in ``rlinf/models/embodiment/openpi/dataconfig/__init__.py``. Keep model architecture and normalization consistent with Stage I.

**Batch size**

The config ``batch_size`` is the micro-batch size before gradient accumulation. Effective batch size is:

.. math::

   \text{True_Batch_Size} = \frac{\text{Global_Batch_Size} \times \text{Input_Batch}}{\text{Micro_Batch_Size} \times \text{Num_GPUs}}

See :doc:`./pi0` for ``global_batch_size`` and ``micro_batch_size`` settings.

**Run**

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_co_training_openpi_pi05


Visualization and results
-------------------------

**TensorBoard**

.. code:: bash

   tensorboard --logdir ./logs --port 6006

**Metrics**

Besides standard RL metrics (see :doc:`π₀ and π₀.₅ visualization <pi0>`), co-training adds:

- ``train/ppo_loss``: PPO (RL) loss.
- ``train/sft_loss``: SFT loss on real data.
- ``actor/total_loss``: :math:`\mathcal{L}_{Total} = \mathcal{L}_{RL} + \beta \mathcal{L}_{SFT}`.
- ``train/loss_ratio``: :math:`\frac{\beta \lvert \mathcal{L}_{SFT} \rvert}{\lvert \mathcal{L}_{RL} \rvert}`. If this stays very large (e.g. :math:`> 10^5`), the logger will warn; consider lowering ``sft_loss_weight``.

**Example outcomes**

- After loading Stage I: ~35% zero-shot success in sim.
- After 100 co-training steps: ~50% success in sim.

For real-robot deployment and ablations, see the paper: *RLinf-Co: Reinforcement Learning–Based Sim–Real Co-Training for VLA Models*.
