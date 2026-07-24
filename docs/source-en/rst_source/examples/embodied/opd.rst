OPD for OpenVLA-OFT on LIBERO
=================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

OPD (On-Policy Distillation) distills on the student policy's own on-policy rollouts:
the student interacts with the environment and stores action tokens; after rollout,
the teacher scores those same student action tokens with dense token-level log
probabilities. In RLinf, the teacher does not generate actions during rollout. It
only scores saved ``forward_inputs.action_tokens`` after rollout and is offloaded
immediately afterward to reduce rollout latency and GPU memory pressure.

Overview
--------

This example trains an **OpenVLA-OFT student** with an **OpenVLA-OFT teacher** on
**LIBERO-Spatial** using OPD.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environment
      :text-align: center

      LIBERO-Spatial

   .. grid-item-card:: Algorithm
      :text-align: center

      OPD

   .. grid-item-card:: Model
      :text-align: center

      OpenVLA-OFT student · OpenVLA-OFT teacher

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** download student/teacher checkpoints -> verify ``unnorm_key`` -> launch ``libero_spatial_opd_openvlaoft`` -> watch ``env/success_once`` and OPD actor metrics.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · :doc:`LIBERO setup <libero>` · a working OpenVLA-OFT + LIBERO environment.

Method
------

OPD uses the relative log probability assigned by teacher and student to the same
student action-token sequence:

.. math::

   A_{OPD} = \log \pi_{teacher}(a_{student} \mid s) - \operatorname{stop\_grad}(\log \pi_{student}(a_{student} \mid s))

Implementation details:

- Student rollout is the only source of environment interaction; the teacher does not sample actions.
- After rollout, the teacher computes dense token logprobs over stored ``forward_inputs.action_tokens``.
- OPD loss uses ``loss_mask`` to exclude invalid tokens after success or termination.

Checkpoints
-----------

.. list-table:: **Checkpoints used by LIBERO-Spatial OPD**
   :header-rows: 1
   :widths: 18 42 26 28

   * - Role
     - Hugging Face checkpoint
     - Config field
     - ``unnorm_key``
   * - Student
     - |huggingface| `RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`_
     - ``actor.model.model_path`` / ``rollout.model.model_path``
     - ``libero_130_no_noops_trajall``
   * - Teacher
     - |huggingface| `RLinf/RLinf-OpenVLAOFT-LIBERO-130 <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130>`_
     - ``rollout.expert_model.model_path``
     - ``libero_130_no_noops_trajall``

Download the checkpoints with either method:

.. code-block:: bash

   # Method 1: git-lfs
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130

   # Method 2: huggingface-hub (set HF_ENDPOINT=https://hf-mirror.com in mainland China)
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130 --local-dir RLinf-OpenVLAOFT-LIBERO-130

Configuration
-------------

The example config is ``examples/embodiment/config/libero_spatial_opd_openvlaoft.yaml``.
Key fields are:

.. code-block:: yaml

   algorithm:
     adv_type: opd
     loss_type: opd
     loss_agg_func: token-mean

   actor:
     global_batch_size: 4096
     micro_batch_size: 32
     model:
       model_path: /path/to/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
       unnorm_key: libero_130_no_noops_trajall
     optim:
       lr: 2.0e-6

   rollout:
     model:
       model_path: /path/to/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
     expert_model:
       model_path: /path/to/RLinf-OpenVLAOFT-LIBERO-130
       unnorm_key: libero_130_no_noops_trajall

   env:
     train:
       rollout_epoch: 8
       max_episode_steps: 240
       max_steps_per_rollout_epoch: 240

.. note::

   Student and teacher must use the same ``unnorm_key``. Both checkpoints in this
   recipe use ``libero_130_no_noops_trajall``.

Run
---

Launch the full training run:

.. code-block:: bash

   source switch_env openvla-oft
   bash examples/embodiment/run_embodiment.sh libero_spatial_opd_openvlaoft

Logs are written under ``runner.logger.log_path`` and can be inspected with TensorBoard:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

Key metrics:

- ``env/success_once``: unnormalized episodic success rate.
- ``actor/opd_reverse_kl``: teacher-vs-student logprob gap on student action tokens.
- ``actor/opd_reward``: token-level distillation reward used by OPD.
- ``actor/policy_loss`` and ``actor/total_loss``: actor update stability.

Results
-------

The table below reports LIBERO-Spatial results. The base student is
|huggingface| `RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`_;
the teacher is |huggingface| `RLinf/RLinf-OpenVLAOFT-LIBERO-130 <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130>`_.

.. list-table:: **LIBERO-Spatial success rate**
   :header-rows: 1
   :widths: 40 24 36

   * - Model / training stage
     - Success rate
     - Notes
   * - OpenVLA-OFT LIBERO-130 Base-Lora student
     - ~67%
     - Student baseline before OPD training.
   * - OPD training, 30 steps
     - 96%+
     - Early OPD result with ``libero_spatial_opd_openvlaoft``.
