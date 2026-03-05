Quickstart 1: PPO Training of VLAs on Maniskill3
=================================================

This quick-start walks you through training the Visual-Language-Action model, including
`OpenVLA <https://github.com/openvla/openvla>`_ and `OpenVLA-OFT <https://github.com/moojink/openvla-oft>`_ on the
`ManiSkill3 <https://github.com/haosulab/ManiSkill>`_ environment with **RLinf**.

Environment Introduction
--------------------------

ManiSkill3 is a GPU-accelerated simulation platform for robotics research,  
focusing on complex contact manipulation and embodied intelligence tasks.  
The benchmark covers multiple domains, including robotic arms, mobile manipulators, humanoid robots, and dexterous hands,  
supporting various tasks such as grasping, assembling, drawing, and locomotion.

We have also implemented system-level optimizations for the GPU simulator (see :doc:`../tutorials/mode/hybrid`).

Launch Training
--------------------------

**Step 1: Download pre-trained models**

If using the **OpenVLA** model, run the following command:

.. code-block:: bash

   # Download OpenVLA pre-trained model
   hf download gen-robot/openvla-7b-rlvla-warmup \
   --local-dir /path/to/model/openvla-7b-rlvla-warmup/

This model is cited in the paper: `paper <https://arxiv.org/abs/2505.19789>`_

If using the **OpenVLA-OFT** model, run the following command:

.. code-block:: bash

   # Download OpenVLA-OFT pre-trained model
   hf download RLinf/Openvla-oft-SFT-libero10-trajall \
   --local-dir /path/to/model/Openvla-oft-SFT-libero10-trajall/
   
   # Download LoRA fine-tuned checkpoint on maniskill
   hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora \
   --local-dir /path/to/model/oft-sft/lora_004000

   # Download assets
   hf download --repo-type dataset RLinf/maniskill_assets \
   --local-dir ./rlinf/envs/maniskill/assets



**Step 2: Modify the configuration file**

Before running the script, you need to modify the ``./examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml`` file according to the download paths of the model and dataset. Specifically, update the following configurations to the path where the `gen-robot/openvla-7b-rlvla-warmup` checkpoint is located.

- ``rollout.model.model_path``  
- ``actor.model.model_path``  



For **OpenVLA-OFT**, modify the ``maniskill_ppo_openvlaoft_quickstart.yaml`` file. Set the following model configuration items to the path where the `RLinf/Openvla-oft-SFT-libero10-trajall` checkpoint is located. At the same time, set the LoRA path to the path where the `RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora` checkpoint is located.

- ``rollout.model.model_path``  
- ``actor.model.model_path``  
- ``actor.model.lora_path``
- ``actor.model.is_lora: True``

**Step 3: Launch training**

After completing the above configuration file modifications, run the following command to launch training:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

.. note::
   If you installed **RLinf** via Docker image (see :doc:`./installation`), please ensure you have switched to the Python environment corresponding to the target model.
   The default environment is ``openvla``.
   If using OpenVLA-OFT or openpi, please use the built-in script `switch_env` to switch environments:
   ``source switch_env openvla-oft`` or ``source switch_env openpi``.

   If you installed **RLinf** in a custom environment, please ensure you have installed the dependencies for the corresponding model, see :doc:`./installation` for details.

For **OpenVLA-OFT**:

.. code-block:: bash

   source switch_env openvla-oft
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft_quickstart


View Training Results
--------------------------

- Final model and metrics save path: ``./logs``  
- Launch as follows:

  .. code-block:: bash

     tensorboard --host 0.0.0.0 --logdir path/to/tensorboard/

After opening TensorBoard, you will see an interface similar to the figure below.  
It is recommended to focus on the following metrics:

- ``rollout/env_info/return``  
- ``rollout/env_info/success_once``  

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/embody-quickstart-metric.jpg" width="800"/>

.. note::
   If you want to specify GPU usage,
   you can modify the parameter  
   ``cluster.component_placement`` in the configuration file.

   Set this item to **0-3** or **0-7** to use 4/8 GPUs according to your actual resources.
   See :doc:`../tutorials/user/yaml` for more detailed instructions on Placement configuration.

   .. code-block:: yaml

      cluster:
      num_nodes: 1
      component_placement:
         actor,env,rollout: 0-3
