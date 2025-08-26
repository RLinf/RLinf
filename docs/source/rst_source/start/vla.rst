Quickstart 1: PPO Training of VLAs on Maniskill3
=================================================

This quick-start walks you through training the Visual-Language-Action model, including
`OpenVLA <https://github.com/openvla/openvla>`_ on the
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

**Step 2: Execute the provided launch script:**

For user convenience, our configuration file is set up to run with a single GPU by default.  
However, if you have multiple GPUs and wish to accelerate the quickstart process,  
we highly recommend updating the following configuration option in  
``./examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml``:  
``cluster.num_gpus_per_node``.

You can dynamically set it to **1, 2, 4, or 8** depending on your available resources.

.. code-block:: yaml

   cluster:
     num_nodes: 1
     num_gpus_per_node: 1
     component_placement:
        actor,rollout: all

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

   <img src="https://github.com/user-attachments/assets/90269207-b638-478b-bf5e-95bd8e2bfb36" width="800"/>




TODO: update the pics for 10 epochs