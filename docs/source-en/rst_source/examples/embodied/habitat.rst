RL with Habitat Benchmark
=========================

This document provides a comprehensive guide to launching and managing the Vision-Language-Navigation Models (VLNs) training task within the RLinf framework, focusing on finetuning a VLN model in the `Habitat <https://aihabitat.org/>`_ environment.

The primary objective is to develop a model capable of performing robotic navigation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (navigation control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Environment
-----------

**Habitat Environment**

- **Environment**: A high-fidelity navigation simulator built on Habitat-Sim
- **Task**: Vision-and-Language Navigation (VLN)
- **Agent**: An embodied agent navigating in Matterport3D environments
- **Observation**: Egocentric RGB images captured from an onboard camera
- **Action Space**: Discrete navigation actions including ``[move_forward, turn_left, turn_right, stop]``

**Data Structure**

- **Images**: RGB observations of shape ``[batch_size, H, W, 3]``
- **Task Descriptions**: Natural-language navigation instructions
- **Actions**: Discrete navigation actions
- **Metrics**: Success rate, SPL, trajectory length, and navigation error

Dependency Installation
--------------------------------------

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --env habitat
   source .venv/bin/activate

   # NaVid must use transformers==4.31.0
   uv pip install transformers==4.31.0


VLN-CE Dataset Preparation
--------------------------

Download the scene dataset:

- For **R2R**, **RxR**: Download the MP3D scenes from `Matterport3D <https://niessner.github.io/Matterport>`_
  official website and put them into the ``VLN-CE/scene_dataset`` folder.

Download the VLN-CE episodes:

- `r2r <https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view>`_
  (Rename ``R2R_VLNCE_v1-3_preprocessed/`` -> ``r2r/``)
- `rxr <https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view>`_
  (Rename ``RxR_VLNCE_v0/`` -> ``rxr/``)

Put them into the ``VLN-CE/datasets`` folder.

Dataset structure:

.. code:: bash

   VLN-CE
   |-- datasets
   |   |-- r2r
   |   |-- rxr
   `-- scene_dataset
       |-- mp3d

Model Download
--------------

Before starting training, you need to download the corresponding pretrained model:

.. code:: bash

   # Download NaVid model
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/Jzzhang/NaVid

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download Jzzhang/NaVid --local-dir VLN-CE/models/navid_weights/NaVid

   # Download EVA-ViT-G model
   wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth

Put them into the ``VLN-CE/models/navid_weights`` folder. And modify the following fields in the file ``VLN-CE/models/navid_weights/NaVid/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split/config.json``:

.. code:: json

   { 
      "image_processor": "rlinf/models/embodiment/navid/processor/clip-patch14-224",
      "mm_vision_tower": "VLN-CE/models/navid_weights/eva_vit_g.pth",
   }

Dataset structure:

.. code:: bash

   VLN-CE
   |-- datasets
   |   |-- r2r
   |   |-- rxr
   |-- models
   |   |-- navid_weights
   |   |   |-- NaVid
   |   |   |-- eva_vit_g.pth
   `-- scene_dataset
       |-- mp3d

Running the Script
------------------

**1. Key Cluster Configuration**

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

You can flexibly configure the GPU count for env, rollout, and actor components.
Additionally, by setting ``pipeline_stage_num = 2`` in the configuration,
you can achieve pipeline overlap between rollout and env, improving rollout efficiency.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also reconfigure the layout to achieve full sharing,
where env, rollout, and actor components all share all GPUs.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

You can also reconfigure the layout to achieve full separation,
where env, rollout, and actor components each use their own GPUs with no
interference, eliminating the need for offloading functionality.



**2. Configuration Files**

Using Habitat R2R as an example:

- NaVid + GRPO:
  ``examples/embodiment/config/habitat_r2r_grpo_navid.yaml``

**3. Launch Commands**

To evaluate NaVid using in the Habitat environment, run:

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh habitat_r2r_grpo_navid