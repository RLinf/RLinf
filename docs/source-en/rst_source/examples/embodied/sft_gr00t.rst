GR00T-N1.6 Supervised Fine-Tuning (SFT)
======================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
    :width: 16px
    :height: 16px
    :class: inline-icon

This document describes how to perform supervised fine-tuning (SFT) of the GR00T-N1.6 model within the RLinf framework. SFT is typically the first stage before reinforcement learning; it fine-tunes the model on offline datasets to better match downstream task distributions. For GR00T-N1.6 we provide example configs for different datasets to help users get started. Instructions for LoRA fine-tuning will be added later.

Contents
--------

- How to configure general supervised fine-tuning in RLinf
- How to launch training on single-machine or multi-node clusters
- How to monitor and evaluate results


Supported datasets
------------------

RLinf currently supports LeRobot-format datasets.

Currently supported dataset formats include:

- gr00t_n1.6_libero


Training configuration
----------------------

Full example config located at:

- ``examples/sft/config/libero_sft_gr00t_16.yaml``

A general GR00T-N1.6 SFT config example is as follows:

1. Cluster configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

     cluster:
        num_nodes: 1     # number of nodes
        hardware_ranks: [[0]]
        component_placement:    # component → GPU mapping
          actor: all

2. Model configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml
    model:
        model_path: "/path/to/GR00T-N1.6-3B" # change to the actual path of the GR00T-N1.6 model

        model_type: "gr00t_1_6_sft"
        precision: "bf16"
        action_dim: 128
        num_action_chunks: 1
        add_value_head: True
        denoising_steps: 1

3. Dataset configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml
    data:
      train_data_paths: "/path/to/your/libero_spatial_dataset" # change to your training data path


Dependency installation
-----------------------

This section describes the dependencies required for performing SFT training for the OpenPI model. For other models, refer to the "Dependency installation" subsection in their respective example docs.

1. Clone the RLinf repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

     # For faster downloads in certain regions, you can use:
     # git clone https://ghfast.top/github.com/RLinf/RLinf.git
     git clone https://github.com/RLinf/RLinf.git
     cd RLinf

2. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Option 1: Use the Docker image
------------------------------

We recommend running experiments using the prebuilt Docker image.

.. code:: bash

     docker run -it --rm --gpus all \
          --shm-size 20g \
          --network host \
          --name rlinf \
          -v .:/workspace/RLinf \
          rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
          # If you need a mirror for faster image downloads, you can use:
          # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

After entering the container, switch to the appropriate virtual environment using the built-in `switch_env` tool:

.. code:: bash

     source switch_env gr00t_16

Option 2: Build your own environment
------------------------------------

You can also install dependencies directly on your machine/cluster. Example commands:

.. code:: bash

     # To speed up dependency installation in some regions, you can add `--use-mirror` to the install.sh command below

     bash requirements/install.sh embodied --model gr00t_16 --env maniskill_libero
     source .venv/bin/activate

Startup script
--------------

Run the training script:

.. code:: bash

    # return to repo root
    bash examples/sft/run_vla_sft.sh libero_sft_gr00t_16

LeRobot SFT model conversion
---------------------------

To use an SFT-trained model in RLinf for reinforcement learning, the SFT model must be converted to the GR00T-N1.6 format. Steps for conversion:

1. Path configuration
~~~~~~~~~~~~~~~~~~~~~

- The conversion script is at ``rlinf/models/embodiment/gr00t_1_6/convert_to_hf.py``

.. code:: python
     base_model_path = "/path/to/GR00T-N1.6-3B"  ## change to the actual path of the GR00T-N1.6 model
     sft_pt_path = "/path/to/logs/**/gr00t_16_sft_libero/checkpoints/**/actor/model_state_dict/full_weights.pt" ## change to the SFT model weights path
     hf_output_path = "/path/to/output/GR00T-1.6-SFT-LIBERO-HF" ## specify the desired output path

.. code:: bash
     python rlinf/models/embodiment/gr00t_1_6/convert_to_hf.py

Fine-tune results
------------------
.. raw:: html

   <div style="display: flex; justify-content: center; margin: 20px 0;">
     <div style="flex: 0.5; text-align: center;">
       <img src="https://github.com/yangzhongii/misc/blob/main/pic/gr00t_1.6_sft_loss.png" style="width: 100%;"/>
       <p><em>GR00T-N1.6 SFT in LIBERO_Spatial Loss Function</em></p>
     </div>
   </div>
