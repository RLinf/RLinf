Quickstart 1: PPO Training of VLAs on Maniskill3
=================================================

This quick-start walks you through training the Visual-Language-Action model, including
`OpenVLA <https://github.com/openvla/openvla>`_ and `OpenVLA-OFT <https://github.com/moojink/openvla-oft>`_ on the
`ManiSkill3 <https://github.com/haosulab/ManiSkill>`_ environment with **RLinf**.
For maximum simplicity, you can run the following scripts within a single GPU.

Environment Introduction
--------------------------

ManiSkill3 is a GPU-accelerated simulation platform for robotics research, 
focusing on contact-rich manipulation and embodied intelligence. 
The benchmark covers diverse domains including robotic arms, mobile manipulators, humanoids, and dexterous hands, 
with tasks such as grasping, assembling, drawing, and locomotion. 

We have implemented customized system-level optimizations for the GPU-based simulator (see :doc:`../tutorials/mode/hybrid`).

Launch Training
-----------------

**Step 1: Download the pre-trained model**

For **OpenVLA**, run:

.. code-block:: bash

   # Download pre-trained OpenVLA model
   hf download gen-robot/openvla-7b-rlvla-warmup \
   --local-dir /path/to/model/openvla-7b-rlvla-warmup/


the model is cited in `paper <https://arxiv.org/abs/2505.19789>`_

For **OpenVLA-OFT**, run:

.. code-block:: bash

   # Download pre-trained OpenVLA-OFT model
   hf download RLinf/Openvla-oft-SFT-libero10-trajall \
   --local-dir /path/to/model/Openvla-oft-SFT-libero10-trajall/
   
   # Download lora fine-tuned ckpt in maniskill
   hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora \
   --local-dir /path/to/model/oft-sft/lora_004000

**Step 2: Execute the provided launch script:**

.. note:: 
   If you have installed RLinf via the Docker image (see :doc:`./installation`), please make sure you have switched to the right Python environment for the target model.
   The default environment is set to ``openvla``. 
   To switch to OpenVLA-OFT or openpi, use the built-in script `switch_env`: 
   ``source switch_env openvla-oft`` or ``source switch_env openpi``.

   If you have installed RLinf in a custom environment, please ensure that you have installed the model's corresponding dependencies as described in :doc:`./installation`.

For user convenience, our configuration file is set up to run with at least two GPUs by default.  
However, if you have multiple GPUs and wish to accelerate the quickstart process,  
we highly recommend updating the following configuration option in  
``./examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml``:  
``cluster.component_placement``.

You can set it to **0-3** or  **0-7** to use 4/8 GPUs depending on your available resources.
Refer to :doc:`../tutorials/user/yaml` for a more detailed explanation of the placement configuration.

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
        actor,rollout: 0-1

Finally, before running the script, you need to modify the corresponding configuration options in the YAML file according to the download paths of the model and dataset. Specifically, for **OpenVLA** update the following configurations to the path of the `gen-robot/openvla-7b-rlvla-warmup` checkpoint:

- ``rollout.model_dir``  
- ``actor.checkpoint_load_path``  
- ``actor.tokenizer.tokenizer_model``  

For **OpenVLA-OFT**, set the following configurations to the path of the `RLinf/Openvla-oft-SFT-libero10-trajall` checkpoint. And set lora path to the path of `RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora` checkpoint:

- ``rollout.model_dir``  
- ``actor.checkpoint_load_path``  
- ``actor.tokenizer.tokenizer_model``  
- ``actor.model.lora_path``
- ``actor.model.is_lora: True``

After these modifications, launch the following script to start training!

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

**Step 3: View the results:**

* Final checkpoints & metrics: ``../results``

* TensorBoard summaries: ``../results/tensorboard``  
  Launch with:

  .. code-block:: bash

     tensorboard --logdir ../results/tensorboard/ --port 6006


Open TensorBoard, and you should see an interface similar to the one below.  
Key metrics to pay attention to include  
``rollout/env_info/return`` and ``rollout/env_info/success_once``.  

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/embody-quickstart-metric.jpg" width="800"/>