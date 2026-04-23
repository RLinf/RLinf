GR00T-N1.6 Supervised Fine-Tuning (SFT)
==================================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
     :width: 16px
     :height: 16px
     :class: inline-icon

This document describes how to perform Supervised Fine-Tuning (SFT) for the GR00T-N1.6 model within the RLinf framework. SFT is typically used as the first stage before reinforcement learning; by fine-tuning the model on offline datasets, it better adapts to the distribution of downstream tasks. For the GR00T-N1.6 model, we provide example configurations for different datasets to help users get started quickly.

*Note: dedicated instructions for LoRA fine-tuning will be added later.*

Contents include
------------------------------------------------------------------

- How to configure general supervised fine-tuning in RLinf
- How to start training on a single machine or multi-node cluster
- How to monitor and evaluate results

Supported datasets
------------------------------------------------------------------

RLinf currently supports datasets in the LeRobot format.

Currently supported specific dataset formats include:

- ``gr00t_n1.6_libero``

Training configuration
------------------------------------------------------------------

The full example configuration is located at:
``examples/sft/config/libero_sft_gr00t_16.yaml``

A general GR00T-N1.6 SFT configuration example is as follows:

1. Cluster configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

     cluster:
       num_nodes: 1     # number of nodes
       hardware_ranks: [[0]]
       component_placement:    # component → GPU mapping
           actor: all

2. Model configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

     model:
       model_path: "/path/to/GR00T-N1.6-3B" # change to the actual path of the GR00T-N1.6 model
       model_type: "gr00t_1_6_sft"
       precision: "bf16"
       action_dim: 128
       num_action_chunks: 1
       add_value_head: True
       denoising_steps: 1

3. Dataset configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

     data:
       train_data_paths: "/path/to/your/libero_spatial_dataset" # change to your training data path

Dependencies
------------------------------------------------------------------

This section describes the dependency environment required for SFT training of the GR00T-N1.6 model.

1. Clone the RLinf repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

     # To improve download speed in China, you can use:
     # git clone https://ghfast.top/github.com/RLinf/RLinf.git
     git clone https://github.com/RLinf/RLinf.git
     cd RLinf

2. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Use the Docker image**

It is recommended to run experiments using the prebuilt Docker image.

.. code-block:: bash

     docker run -it --rm --gpus all \
           --shm-size 20g \
           --network host \
           --name rlinf \
           -v .:/workspace/RLinf \
           rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
           # If you need a mirror to accelerate downloading the image in China, you can use:
           # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

After entering the container, switch to the corresponding virtual environment using the built-in ``switch_env`` tool:

.. code-block:: bash

     source switch_env gr00t_16

**Option 2: Build your own environment**

.. code-block:: bash

     # To speed up dependency installation in China, you can add `--use-mirror` to the install.sh command below
     bash requirements/install.sh embodied --model gr00t_16 --env maniskill_libero
     source .venv/bin/activate

Model Download
------------------------------------------------------------------
Before starting SFT training, you need to download the corresponding datasets and the pre-trained GR00T-N1.6 model, and place them in appropriate locations.
Currently supports four Libero tasks: Spatial, Object, Goal, 10

Libero Dataset Download:
.. code-block:: bash
     # Method 1: Using git clone
     git lfs install
     git clone https://hf-mirror.com/datasets/ZibinDong/libero_spatial
     # You can also download other datasets such as libero_object, libero_goal, libero_10
     # https://hf-mirror.com/datasets/ZibinDong/libero_object
     # https://hf-mirror.com/datasets/ZibinDong/libero_goal
     # https://hf-mirror.com/datasets/ZibinDong/libero_10

     # Method 2: Using huggingface-hub
     # To improve download speed in China, you can set:
     # export HF_ENDPOINT=https://hf-mirror.com
     pip install huggingface-hub
     hf download ZibinDong/libero_spatial --local-dir Gr00t_16-libero-Spatial-dataset
     # You can also download other datasets such as libero_object, libero_goal, libero_10
     # hf download ZibinDong/libero_object --local-dir Gr00t_16-libero-Object-dataset
     # hf download ZibinDong/libero_goal --local-dir Gr00t_16-libero-Goal-dataset
     # hf download ZibinDong/libero_10 --local-dir Gr00t_16-libero-10-dataset


Launch script
------------------------------------------------------------------

Run the training script:

.. code-block:: bash

     # Run from the repository root
     bash examples/sft/run_vla_sft.sh libero_sft_gr00t_16

LeRobot SFT model format conversion
------------------------------------------------------------------

To use a supervised fine-tuned model in RLinf for reinforcement learning training, the SFT model needs to be converted to the standard GR00T-N1.6 format.

1. Configure paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script location: ``rlinf/models/embodiment/gr00t_1_6/convert_to_hf.py``

.. code-block:: python

     base_model_path = "/path/to/GR00T-N1.6-3B"  ## change to the actual path of the GR00T-N1.6 model
     sft_pt_path = "/path/to/logs/**/gr00t_16_sft_libero/checkpoints/**/actor/model_state_dict/full_weights.pt" ## path to the SFT weights
     hf_output_path = "/path/to/output/GR00T-1.6-SFT-LIBERO-HF" ## specify the output path

2. Execute the conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

     python rlinf/models/embodiment/gr00t_1_6/convert_to_hf.py

Fine-tuning results (SFT)
------------------------------------------------------------------

.. raw:: html

     <div style="display: flex; justify-content: center; margin: 20px 0;">
       <div style="flex: 0.5; text-align: center;">
           <img src="https://github.com/yangzhongii/misc/blob/main/pic/gr00t_1.6_sft_loss.png?raw=true" style="width: 100%;"/>
           <p><em>Loss curve of GR00T-N1.6 SFT on LIBERO_Spatial</em></p>
       </div>
     </div>