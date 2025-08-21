Quickstart 1: PPO Training of VLA on Maniskill3
=================================================

This quick-start walks you through training the Visual-Language-Action model, including
`OpenVLA <https://github.com/openvla/openvla>`_ and
`OpenVLA-OFT <https://github.com/moojink/openvla-oft>`_ on the
`ManiSkill3 <https://github.com/haosulab/ManiSkill>`_ environment with **RLinf**.

Environment Introduction
--------------------------

ManiSkill3 is a GPU-accelerated simulation platform for robotics research, 
focusing on contact-rich manipulation and embodied intelligence. 
The benchmark covers diverse domains including robotic arms, mobile manipulators, humanoids, and dexterous hands, 
with tasks such as grasping, assembling, drawing, and locomotion. 

**Step 1: Download the pre-trained model and prepare the environment:**

.. code-block:: bash

   # Download pre-trained OpenVLA model
   hf download openvla/Openvla-oft-SFT-libero10-trajall \
   --local-dir /storage/download_models/Openvla-oft-SFT-libero10-trajall/

**Step 2: Execute the provided launch script:**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla # openvla model
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft # openvlaoft model

**Step 3: View the results:**

* Final checkpoints & metrics: ``/storage/results``

* TensorBoard summaries: ``/storage/results/tensorboard``  
  Launch with:

  .. code-block:: bash

     tensorboard --logdir /storage/results/tensorboard/ --port 6006

* Training logs: ``./logs``

* Video recordings: ``./logs/video/train`` and ``./logs/video/eval``