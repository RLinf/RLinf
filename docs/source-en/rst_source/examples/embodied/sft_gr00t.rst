GR00T-N1.6 Supervised Fine-tuning Training (SFT)
==================================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document explains how to perform **supervised fine-tuning (SFT)** for the GR00T-N1.6 model within the RLinf framework. SFT is usually the first stage before reinforcement learning, where the model is fine-tuned on an offline dataset to better adapt to the target task distribution. For GR00T-N1.6, we provide sample configurations for different datasets to help users get started quickly.

*Note: A dedicated guide for LoRA fine-tuning will be added later.*

Contents
------------------------------------------------------------------

- How to configure general supervised fine-tuning in RLinf
- How to start training on a single machine or multi-node cluster
- How to monitor and evaluate results

Supported datasets
------------------------------------------------------------------

RLinf currently supports datasets in LeRobot format.

Supported dataset formats include:

- ``gr00t_n1.6_libero``

Training configuration
------------------------------------------------------------------

A full example configuration is available at:
``examples/sft/config/libero_sft_gr00t_n1d6.yaml``

A general GR00T-N1.6 SFT configuration example is shown below:

1. Cluster configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

   cluster:
     num_nodes: 1     # number of nodes
     hardware_ranks: [[0]]
     component_placement:    # component -> GPU mapping
       actor: all

2. Model configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

   model:
     model_path: "/path/to/GR00T-N1.6-3B" # change to the actual GR00T-N1.6 model path
     model_type: "gr00t_n1d6_sft"
     precision: "bf16"
     action_dim: 128
     num_action_chunks: 1
     add_value_head: True
     denoising_steps: 1

3. Dataset configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

   data:
     train_data_paths: "/path/to/your/libero_spatial_dataset" # change to your training dataset path

Dependency installation
------------------------------------------------------------------

This section describes the dependency environment required for GR00T-N1.6 SFT training.

1. Clone the RLinf repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # For faster download in China, you can use:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Use Docker image**

It is recommended to run experiments directly using the pre-built Docker image.

.. code-block:: bash

   docker run -it --rm --gpus all \
       --shm-size 20g \
       --network host \
       --name rlinf \
       -v .:/workspace/RLinf \
       rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
       # If you need accelerated image download in China, you can use:
       # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

After entering the container, switch to the appropriate virtual environment using the built-in ``switch_env`` tool:

.. code-block:: bash

   source switch_env gr00t_n1d6

**Option 2: Build your own environment**

.. code-block:: bash

   # To speed up dependency installation in China, add `--use-mirror` to the install.sh command below
   bash requirements/install.sh embodied --model gr00t_n1d6 --env maniskill_libero
   source .venv/bin/activate

Model download
------------------------------------------------------------------
Before starting SFT training, download the required dataset and the GR00T-N1.6 pretrained model and place them in the appropriate locations.
Currently, four Libero tasks are supported: Spatial, Object, Goal, and 10.

Libero dataset download:

.. code-block:: bash
   
   # Method 1: use git clone
   git lfs install
   git clone https://hf-mirror.com/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
   # You can also download other datasets such as libero_object, libero_goal, libero_long
   # https://hf-mirror.com/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot
   # https://hf-mirror.com/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot
   # https://hf-mirror.com/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot

   # Method 2: use huggingface-hub
   # To speed up downloads in China, set:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir Gr00t_16-libero-Spatial-dataset
   # You can also download other datasets such as libero_object, libero_goal, libero_10
   # hf download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot --repo-type dataset --local-dir Gr00t_16-libero-Object-dataset
   # hf download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot --repo-type dataset --local-dir Gr00t_16-libero-Goal-dataset
   # hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot --repo-type dataset --local-dir Gr00t_16-libero-10-dataset

Data version conversion
Since the provided data version is an older Lerobot version, it needs to be converted to the latest version.

.. code-block:: bash

   python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=Gr00t_16-libero-<task>-dataset --root=/path/to/RLinf --push-to-hub=false --force-conversion
   # e.g.：python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=Gr00t_16-libero-Spatial-dataset  --root=/workspace/test/RLinf  --push-to-hub=false --force-conversion

GR00T-N1.6 model download

.. code-block:: bash

   # Method 1: use git clone
   git lfs install
   git clone https://huggingface.co/nvidia/GR00T-N1.6-3B

   # Method 2: use huggingface-hub
   # To speed up downloads in China, set:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download nvidia/GR00T-N1.6-3B --repo-type model --local-dir GR00T-N1.6-3B

Launch script
------------------------------------------------------------------

Run the training script:

.. code-block:: bash

   # Execute from the repository root
   bash examples/sft/run_vla_sft.sh libero_sft_gr00t_n1d6

LeRobot SFT model format conversion
------------------------------------------------------------------

To use the supervised fine-tuned model for reinforcement learning in RLinf, the SFT model needs to be converted to the standard GR00T-N1.6 format.

Execute conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export REPO_PATH="/path/to/RLinf"

   python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
      --config-path /path/to/RLinf/rlinf/utils/ckpt_convertor/fsdp_convertor/config \
      --config-name fsdp_model_convertor \
      convertor.ckpt_path="/path/to/RLinf/logs/yymmdd-hours:minutes:seconds/gr00t_n1d6_sft_libero/checkpoints/global_step_<>/actor/model_state_dict/full_weights.pt" \
      convertor.save_path="/path/to/where/you/put/GR00T-1.6-SFT-LIBERO-Spaial-HF" \
      ++model.model_type="gr00t_n1d6_sft" \
      ++model.model_path="/path/to/official/GR00T-N1.6-3B" \
      ++model.embodiment_tag="libero_panda" \
      ++model.denoising_steps=4 \
      ++model.num_action_chunks=1 \
      ++model.obs_converter_type="libero" \
      ++model.is_lora=false \
      ++model.rl_head_config.add_value_head=false \
      ++model.rl_head_config.disable_dropout=true

Fine-tuning results display (SFT)
------------------------------------------------------------------

.. raw:: html

   <div style="display: flex; justify-content: center; margin: 20px 0;">
     <div style="flex: 0.5; text-align: center;">
       <img src="https://github.com/RLinf/misc/blob/main/pic/gr00t_1.6_sft_loss.png?raw=true" style="width: 100%;"/>
       <p><em>GR00T-N1.6 SFT loss curve on LIBERO_Spatial</em></p>
     </div>
   </div>
