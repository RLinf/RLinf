DiffusionART
============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   SD3 and Wan2.2 generation models trained with DiffusionART.

Fine-tune SD3 or Wan2.2 with DiffusionART to improve text rendering in images
and videos using OCR rewards. Complete the :doc:`shared setup <video_models>`,
then launch the image or video config.

For Pi0 robot policies, see the :ref:`LIBERO Object and Spatial recipes <diffusion-art-libero>`.

Overview
--------

Both configs use LoRA, FSDP2, and deterministic ODE sampling. OCR rewards measure
how closely the generated text matches the target text in the prompt.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      DiffusionART

   .. grid-item-card:: Models
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: Environments / Data
      :text-align: center

      OCR prompts

   .. grid-item-card:: Training
      :text-align: center

      LoRA · FSDP2 · BF16

Tasks
~~~~~

Choose the config for image or video generation.

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - Model
     - Config
     - Task
   * - SD3.5 Medium
     - ``sd3_art_ocr``
     - Image OCR
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_art_video_ocr``
     - Text-to-video OCR

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Each generated sample receives one OCR score. For videos, the reward is the
mean score across sampled frames.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Value
     - Meaning
   * - Observation
     - A text prompt containing quoted target text.
   * - Action
     - A generated image or video.
   * - Reward
     - An OCR similarity score in ``[0, 1]``.

How ART Works
-------------

ART combines each prompt group's rewards and final latents into a shared
update direction. Training uses denoising states from the generated trajectories.
Both update windows share velocity targets frozen before the first update.

For the derivation and reference implementation, see the
`DiffusionART paper <https://arxiv.org/abs/2610.12572>`_,
`official code <https://github.com/vanzll/DiffusionART>`_, and
`project page <https://vanzll.github.io/diffusionart-project/>`_.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run
``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt
images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.
Both ART configs use CPU OCR; see :doc:`video_models` for full shared setup notes.

.. tab-set::

   .. tab-item:: SD3.5 Medium (Image OCR)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion && source .venv/bin/activate

      **2. Download**

      Accept the SD3.5 Medium access terms on Hugging Face first, then download the checkpoint and OCR prompts:

      .. code-block:: bash

         hf download stabilityai/stable-diffusion-3.5-medium --local-dir ./stable-diffusion-3.5-medium
         mkdir -p ./ocr_data
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt -o ./ocr_data/train.txt
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt -o ./ocr_data/test.txt

      **3. Launch**

      Set ``actor.model.model_path``, ``env.train.dataset.path``, and ``env.eval.dataset.path`` in ``examples/diffusion/config/sd3_art_ocr.yaml`` (adjust GPU placement with ``cluster.component_placement``), then launch:

      .. code-block:: bash

         bash examples/diffusion/run_diffusion.sh sd3_art_ocr

   .. tab-item:: Wan2.2 TI2V 5B (Video OCR)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./Wan2.2-TI2V-5B-Diffusers
         mkdir -p ./ocr_data
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt -o ./ocr_data/train.txt
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt -o ./ocr_data/test.txt

      **3. Launch**

      Set ``actor.model.model_path``, ``env.train.dataset.path``, and ``env.eval.dataset.path`` in ``examples/diffusion/config/wan22_ti2v_5b_art_video_ocr.yaml`` (adjust GPU placement with ``cluster.component_placement``), then launch:

      .. code-block:: bash

         bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr

Logs go under ``logs/<timestamp>-<config>/``. Training reports image OCR scores
as ``env/avg`` and video OCR scores as ``env/video_ocr``. For scalar definitions,
see :doc:`Training Metrics </rst_source/reference/metrics>`.

Configure ART Updates
~~~~~~~~~~~~~~~~~~~~~

Each round generates ``env.train.total_num_envs * env.train.rollout_epoch``
samples. The default ``algorithm.art_updates_per_rollout=2`` splits each round
into two updates, so ``actor.global_batch_size`` is half the samples per round.
For example, 96 environments and 12 rollout epochs give 1152 samples per round
and a global batch size of 576.

The main update settings are:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Setting
     - Effect
   * - ``algorithm.art_timestep_fraction``
     - Fraction of denoising states used for training: ``0.5`` for SD3, ``0.25`` for Wan.
   * - ``algorithm.art_guidance_gain``
     - Reward guidance scale; defaults to ``1.0``.
   * - ``algorithm.art_corrective_mask_enabled``
     - Enables the corrective movement mask; defaults to ``True``.
   * - ``algorithm.art_mask_threshold``
     - Movement threshold for the mask; defaults to ``1e-5`` with the ``sde_kl`` metric.

.. _diffusion-art-results:

Expected Results
----------------

The experiments use four NVIDIA A800 80 GB GPUs, Wan2.2 TI2V 5B, LoRA
(rank ``32``, alpha ``64``), FSDP2, BF16, and CPU OCR. Generation uses
512 × 512 resolution, five frames, ten denoising steps, and guidance scale ``1.0``.

ART uses the ``wan22_ti2v_5b_art_video_ocr`` config; NFT uses
``wan22_ti2v_5b_nft_video_ocr``. Training reward is the mean ``env/video_ocr``
over the last 5760 trajectories. For repeated runs, the table reports the mean
± sample standard deviation.

The first comparison uses ``video_level`` rewards for ART and ``frame_level``
rewards for NFT.

.. list-table:: Wan Video OCR at 40,320 Generated Trajectories
   :header-rows: 1
   :widths: 24 8 16 12 20 20

   * - Method
     - Runs
     - Trajectories per Round
     - Learning Rate
     - Training Reward
     - Training Time (h)
   * - ART
     - 1
     - 1152
     - 1e-4
     - 0.8584
     - 10.01
   * - NFT
     - 2
     - 576
     - 4e-4
     - 0.7565 ± 0.0544
     - 11.29 ± 0.52

With these configurations, ART's training reward at 40,320 trajectories is
**13.5% higher** and its training time **11.3% shorter** than the NFT averages. Training time comes
from the training-loop timer.

The second comparison matches trajectories per round and learning rate.
Both methods use ``video_level`` rewards, with one run each. The table compares
the first 29 rounds of each run.

.. list-table:: Wan Video OCR at 33,408 Generated Trajectories
   :header-rows: 1
   :widths: 18 22 18 22 20

   * - Method
     - Trajectories per Round
     - Learning Rate
     - Training Reward
     - Training Time (h)
   * - ART
     - 1152
     - 1e-4
     - 0.8446
     - 8.31
   * - NFT
     - 1152
     - 1e-4
     - 0.1076
     - 8.23

To reproduce the experiments, set the model and data paths in the chosen
config and apply these shared settings:

.. dropdown:: Advanced configuration (optional)

   .. code-block:: yaml

      cluster:
        component_placement:
          actor,env,rollout: "0-3"
      algorithm:
        group_size: 8
      actor:
        seed: 42
        model:
          wan22_ti2v_5b:
            max_generation_batch_size: 2
            compile_transformer_forward: True
            offload_auxiliary_modules: True
      env:
        train:
          total_num_envs: 96
          seed: 42
          reward:
            use_gpu: False
        eval:
          seed: 42
          reward:
            use_gpu: False
      runner:
        val_check_interval: -1

The batch and step settings for each configuration are below. Both NFT
configurations use ``algorithm.nft_tau=[1.0,0.01,0,70]``.

.. list-table:: Training Configurations
   :header-rows: 1
   :widths: 46 18 18 18

   * - Setting
     - ART, 1152/Round
     - NFT, 576/Round
     - NFT, 1152/Round
   * - ``env.train.rollout_epoch``
     - 12
     - 6
     - 12
   * - ``actor.global_batch_size``
     - 576
     - 288
     - 576
   * - ``actor.micro_batch_size``
     - 2
     - 1
     - 2
   * - ``actor.optim.lr``
     - 1e-4
     - 4e-4
     - 1e-4
   * - ``env.train.reward.reward_type``
     - ``video_level``
     - ``frame_level``
     - ``video_level``
   * - ``runner.max_epochs`` / ``runner.max_steps``
     - 70
     - 70
     - 200
   * - ``runner.save_interval``
     - 10
     - -1
     - 10

.. _diffusion-art-libero:

LIBERO Object and Spatial
-------------------------

Train Pi0 on complete LIBERO trajectories with natural-anchor or state-cluster grouping. Both subtract a state-dependent anchor before computing action–return covariance. The anchor averages two deterministic denoising runs from shared reference noise and its negation; it leaves the sampled actions unchanged.

.. list-table:: LIBERO ART Recipes
   :header-rows: 1

   * - Suite
     - Natural Anchor
     - State Clusters
   * - Object
     - ``libero_object_0_art_openpi``
     - ``libero_object_0_art_cluster_openpi``
   * - Spatial
     - ``libero_spatial_0_art_openpi``
     - ``libero_spatial_0_art_cluster_openpi``

The recipes default to task 0, 128 training environments, four denoising steps, learning rate 5e-7, FP32 master parameters, and 50 training rounds. Episodes last up to 240 environment steps, with five actions per decision. Two updates use disjoint trajectory batches of 3,072 decision rows each; rows after termination are masked. Guidance, restoration, and the trust-region mask use only executed action coordinates.

Choose a Grouping Method
~~~~~~~~~~~~~~~~~~~~~~~~

**Natural anchor** (``algorithm.art_grouping=index``) compares eight trajectories with the same reset group and decision index. They start from matching simulator initializations and sample independent actions. Later decisions train on diverged states; subtracting the anchor reduces differences in the states' mean actions.

**State clusters** (``algorithm.art_grouping=cluster``) group nearby live states across actor ranks. Each cluster shares a task and contact phase, contains four to eight members, and includes at most one decision per trajectory. The first decision keeps the exact reset groups; unclustered later rows are masked. Features cover relative gripper/object/goal positions, full orientations, finger opening, and remaining time. ``algorithm.art_cluster_radius=1.0`` specifies a normalized feature-space radius.

Run Training
~~~~~~~~~~~~

Install the OpenPi/LIBERO environment with :doc:`../start/installation` and use four visible GPUs. Download the SFT checkpoint, including its normalization statistics:

.. tab-set::

   .. tab-item:: Hugging Face

      .. code-block:: bash

         hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
           --local-dir ./RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

In ``examples/embodiment/config/libero_object_0_art_openpi.yaml``, set ``actor.model.model_path`` to the downloaded directory. All four recipes inherit this path, and rollout uses the actor's checkpoint. Set the runtime environment:

.. code-block:: bash

   export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false

Train Object task 0 with natural-anchor grouping:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_object_0_art_openpi LIBERO

To reproduce the Spatial task 6 cluster run, add the following overrides to ``examples/embodiment/config/libero_spatial_0_art_cluster_openpi.yaml``:

.. code-block:: yaml

   env:
     train:
       task_id_filter: [6]
     eval:
       task_id_filter: [6]

Launch the selected recipe:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_0_art_cluster_openpi LIBERO

Change both task filters to run another task, or choose the natural-anchor recipe from the table. To reproduce the joint ten-task results, set both filters to ``null`` in the selected Spatial recipe and merge these settings into its existing YAML sections:

.. code-block:: yaml

   runner:
     max_steps: 200
     max_epochs: 200
   env:
     train:
       task_id_filter: null
     eval:
       task_id_filter: null
       max_steps_per_rollout_epoch: 2640

Launch natural-anchor training with ``bash examples/embodiment/run_embodiment.sh libero_spatial_0_art_openpi LIBERO``, or use ``libero_spatial_0_art_cluster_openpi`` for clusters. Evaluation runs every ten rounds. Logs are written under ``logs/``.

Results
~~~~~~~

The runs use four GPUs, actor seed 42, and seed 0 for both training and evaluation environments. Success is measured by ``eval/success_once``. Single-task runs evaluate 50 trajectories per checkpoint; joint Spatial runs evaluate 500. SFT is the step-0 evaluation of the initial policy in the selected run.

All single-task runs train for 50 rounds. For each task and grouping method, the tables report the completed run with the highest round-50 success rate and that run’s SFT baseline.

.. list-table:: Natural-Anchor Single-Task Success Rates
   :header-rows: 1

   * - Task
     - SFT
     - ART, Round 50
     - Gain (pp)
   * - Object 0
     - 32%
     - 100%
     - +68
   * - Spatial 0
     - 90%
     - 94%
     - +4
   * - Spatial 1
     - 84%
     - 100%
     - +16
   * - Spatial 2
     - 100%
     - 100%
     - +0
   * - Spatial 3
     - 74%
     - 94%
     - +20
   * - Spatial 4
     - 66%
     - 88%
     - +22
   * - Spatial 5
     - 2%
     - 0%
     - -2
   * - Spatial 6
     - 62%
     - 100%
     - +38
   * - Spatial 7
     - 50%
     - 86%
     - +36
   * - Spatial 8
     - 68%
     - 88%
     - +20
   * - Spatial 9
     - 20%
     - 68%
     - +48

.. list-table:: State-Cluster Single-Task Success Rates
   :header-rows: 1

   * - Task
     - SFT
     - ART, Round 50
     - Gain (pp)
   * - Spatial 6
     - 56%
     - 96%
     - +40
   * - Spatial 7
     - 50%
     - 78%
     - +28
   * - Spatial 8
     - 68%
     - 86%
     - +18
   * - Spatial 9
     - 18%
     - 38%
     - +20

Each joint run trains one policy on all ten Spatial tasks for 200 rounds. The table shows the final success rate, the highest success rate during training, and the round at which it occurs.

.. list-table:: Joint Ten-Task Spatial Success Rates
   :header-rows: 1

   * - Grouping
     - SFT
     - ART, Round 200
     - Gain (pp)
     - Best ART
     - Best Round
   * - Natural anchor
     - 61.4%
     - 74.0%
     - +12.6
     - 75.0%
     - 190
   * - State clusters
     - 61.4%
     - 72.6%
     - +11.2
     - 77.6%
     - 170

Watch ``eval/success_once`` in TensorBoard. For clusters, ``actor/cluster_row_frac`` reports coverage and ``actor/cluster_active_frac`` reports the fraction of clusters with nonzero guidance. See :doc:`../reference/metrics` for training metrics and :doc:`../evaluations/index` for standalone evaluation.
