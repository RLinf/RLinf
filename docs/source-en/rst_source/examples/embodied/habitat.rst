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

   # NaVid must require transformers==4.31.0
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

   # Download CMA model
   # Method 1: Using gdown
   # This link contains the ckpt of the CMA model.
   gdown https://drive.google.com/uc?id=1o9PgBT38BH9pw_7V1QB3XUkY8auJqGKw

   # Method 2: Using wget
   # This link contains the depth encoder pretrained weights.
   wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip
   # This link contains the dataset which is preprocessed and instruction encoder embedding.
   wget https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view
   # The rgb encoder pretrained weights is official resnet50 weights.
   # It's location is RLinf/rlinf/models/embodiment/cma/modules/resnet_encoders.py:186

Put them into the ``VLN-CE/models/cma_weights`` folder.Then use the following script to convert the checkpoint to a ``best_ckpt_r2r.pth``:

.. code:: python

   import torch
   import pickle

   # ========== Modify the path here ==========
   CKPT_PATH = "/path/to/your/checkpoint.pth"
   OUTPUT_PATH = "/path/to/output/best_ckpt_r2r.pth"
   class Placeholder:
      def __init__(self, *args, **kwargs):
         self.__dict__ = {}
      def __setitem__(self, k, v):
         self.__dict__[k] = v
      def __setstate__(self, state):
         self.__dict__ = state if isinstance(state, dict) else {}
   class SafeUnpickler(pickle.Unpickler):
      def find_class(self, module, name):
         if 'habitat' in module.lower() or 'vlnce' in module.lower():
               return Placeholder
         try:
               return super().find_class(module, name)
         except:
               return Placeholder
   class SafePickleModule:
      Unpickler = SafeUnpickler
   SafePickleModule.__name__ = 'pickle'

   checkpoint = torch.load(CKPT_PATH, map_location="cpu", pickle_module=SafePickleModule)
   state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint.get("model_state_dict") or checkpoint
   torch.save(state_dict, OUTPUT_PATH)
   print(f"successfully saved to {OUTPUT_PATH}")

Dataset structure:

.. code:: bash

   VLN-CE
   |-- datasets
   |   |-- r2r
   |   |-- rxr
   |-- models
   |   |-- cma_weights
   |   |   |-- ckpt
   |   |   |-- ddppo-models
   |   |   |-- instruction_encoder_embedding
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

- CMA + GRPO:
  ``examples/embodiment/config/habitat_r2r_grpo_cma.yaml``

**3. Launch Commands**

To evaluate NaVid using in the Habitat environment, run:

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh habitat_r2r_grpo_cma