Evaluation 1: Embodied Agent Scenario
========================================

Introduction
------------
RLinf ships with **turn-key evaluation scripts** to benchmark a trained
embodied agent in both *in-distribution* and *out-of-distribution*
settings.  
Two simulation suites are currently supported:

- `ManiSkill3 <https://github.com/haosulab/ManiSkill>`_
- `LIBERO <https://github.com/Lifelong-Robot-Learning/LIBERO>`_

All helper scripts reside in ``examples/embodiment/``;
only the obvious placeholders (checkpoint paths, GPU IDs, etc.) must be
edited before running.


Quick Start — ManiSkill3
------------------------

**Full launch script**

.. code-block:: bash

   #! /bin/bash
   export HF_ENDPOINT=https://hf-mirror.com

   export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
   export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
   export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

   export CUDA_LAUNCH_BLOCKING=1
   export HYDRA_FULL_ERROR=1

   EVAL_NAME=grpo-openvlaoft
   CKPT_PATH=YOUR_CKPT_PATH           # Optional: .pt file or None, if None, will use the checkpoint in rollout.model.model_path
   CONFIG_NAME=YOUR_CFG_NAME          # env.eval must be maniskill_ood_template
   TOTAL_NUM_ENVS=YOUR_TOTAL_NUM_ENVS # total number of evaluation environments
   EVAL_ROLLOUT_EPOCH=YOUR_EVAL_ROLLOUT_EPOCH # eval rollout epoch, total_trajectory_num = eval_rollout_epoch * total_num_envs
   for env_id in \
       "PutOnPlateInScene25VisionImage-v1" "PutOnPlateInScene25VisionTexture03-v1" \
       "PutOnPlateInScene25VisionTexture05-v1" "PutOnPlateInScene25VisionWhole03-v1"  \
       "PutOnPlateInScene25VisionWhole05-v1" "PutOnPlateInScene25Carrot-v1"           \
       "PutOnPlateInScene25Plate-v1" "PutOnPlateInScene25Instruct-v1"                 \
       "PutOnPlateInScene25MultiCarrot-v1" "PutOnPlateInScene25MultiPlate-v1"         \
       "PutOnPlateInScene25Position-v1" "PutOnPlateInScene25PositionChangeTo-v1" ;    \
   do
       obj_set="test"
       LOG_DIR="${REPO_PATH}/logs/eval/${EVAL_NAME}/$(date +'%Y%m%d-%H:%M:%S')-${env_id}-${obj_set}"
       MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
       mkdir -p "${LOG_DIR}"
       CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ \
            --config-name ${CONFIG_NAME} \
            runner.logger.log_path=${LOG_DIR} \
            algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH} \
            env.eval.total_num_envs=${TOTAL_NUM_ENVS} \
            env.eval.init_params.id=${env_id} \
            env.eval.init_params.obj_set=${obj_set} \
            runner.eval_policy_path=${CKPT_PATH}"
       echo ${CMD}  > "${MEGA_LOG_FILE}"
       ${CMD} 2>&1 | tee -a "${MEGA_LOG_FILE}"
   done

   for env_id in \
       "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25MultiCarrot-v1" \
       "PutOnPlateInScene25MultiPlate-v1" ; \
   do
       obj_set="train"
       LOG_DIR="${REPO_PATH}/logs/eval/${EVAL_NAME}/$(date +'%Y%m%d-%H:%M:%S')-${env_id}-${obj_set}"
       MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
       mkdir -p "${LOG_DIR}"
       CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ \
            --config-name ${CONFIG_NAME} \
            runner.logger.log_path=${LOG_DIR} \
            algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH} \
            env.eval.init_params.id=${env_id} \
            env.eval.total_num_envs=${TOTAL_NUM_ENVS} \
            env.eval.init_params.obj_set=${obj_set} \
            runner.eval_policy_path=${CKPT_PATH}"
       echo ${CMD}  > "${MEGA_LOG_FILE}"
       ${CMD} 2>&1 | tee -a "${MEGA_LOG_FILE}"
   done

The script first evaluates twelve **OOD** tasks, then three
**ID** tasks, and writes each log to
``logs/eval/<EVAL_NAME>/…/run_ppo.log``.


Quick Start — LIBERO
--------------------

**Full launch script**

.. code-block:: bash

   #! /bin/bash

   export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
   export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
   export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

   export MUJOCO_GL="osmesa"
   export PYOPENGL_PLATFORM="osmesa"
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

   # path to the LIBERO repo
   export LIBERO_PATH="/opt/LIBERO"
   export PYTHONPATH=${LIBERO_PATH}:$PYTHONPATH

   export CUDA_LAUNCH_BLOCKING=1
   export HYDRA_FULL_ERROR=1

   CONFIG_NAME=${1:-libero_goal_grpo_openvlaoft.eval}

   LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')"
   MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
   mkdir -p "${LOG_DIR}"

   CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ \
        --config-name ${CONFIG_NAME} \
        runner.logger.log_path=${LOG_DIR}"

   echo ${CMD}
   ${CMD} 2>&1 | tee "${MEGA_LOG_FILE}"

**Model-path settings**

.. code-block:: yaml

   rollout:
     model:
       model_path: "/path/to/sft_base_model/"
   actor:
     model:
       model_path: "/path/to/sft_base_model/"
     tokenizer:
       tokenizer_model: "/path/to/sft_base_model/"

**Key YAML fields**

Main YAML + ``config/env/eval/libero_goal.yaml``:

==========================  =============================================
Field                       Purpose
==========================  =============================================
``simulator_type``          Must be ``libero``
``task_suite_name``         LIBERO split (e.g. ``libero_goal``)
``max_episode_steps``       Episode horizon (default 512)
``seed``                    Environment seed
``num_envs``                Parallel evaluation episodes (e.g. 500)
==========================  =============================================

**Possible Issues**

In the latest RLinf code, training and rollout will not cause errors, but if you use intermediate models trained with early RLinf code (GRPO algorithm) and run it in the new version RLinf framework code, you may encounter an error where the loaded model contains extra keys (keys starting with ``value_head.`` ), for example:

.. code-block:: console

   RuntimeError: Error(s) in loading state_dict for OpenVLAOFTForRLActionPrediction:
	Unexpected key(s) in state_dict: "value_head.head_l1.weight", "value_head.head_l1.bias", "value_head.head_l2.weight", "value_head.head_l2.bias", "value_head.head_l3.weight".

You can modify the code at the end of the ``rlinf/models/__init__.py`` file (in the ``get_model`` function). Change:

.. code-block:: python

   if hasattr(cfg, "ckpt_path") and cfg.ckpt_path is not None:
        model_dict = torch.load(cfg.ckpt_path)
        model.load_state_dict(model_dict)
    return model

to:

.. code-block:: python

   if hasattr(cfg, "ckpt_path") and cfg.ckpt_path is not None:
        model_dict = torch.load(cfg.ckpt_path)
        filtered_dict = {k: v for k, v in model_dict.items() if not k.startswith('value_head')}
        model.load_state_dict(filtered_dict, strict=False)
    return model

After modification, the command can run normally.


Results
-------

Both launch scripts end with a **summary line** in the logs:

.. code-block:: javascript

   eval_metrics={
       'eval/env_info/success_once': 0.8984375,
       'eval/env_info/return': 1.0476562,
       'eval/env_info/episode_len': 80.0,
       'eval/env_info/reward': 0.0130957,
       'eval/env_info/success_at_end': 0.859375
   }

``success_once`` is the **success rate** (task succeeded at least once
within an episode).  
Metrics are also written to TensorBoard if enabled.


Environments
------------

.. list-table:: Supported Embodied-Agent Suites
   :header-rows: 1
   :widths: 20 80

   * - Environment
     - Brief Description
   * - ``ManiSkill3``
     - A high-fidelity SAPIEN-based benchmark covering diverse
       manipulation skills (grasp, place, push).  Evaluation focuses on
       **Put-on-Plate** tasks with multiple OOD texture/object splits.
   * - ``LIBERO``
     - A large-scale benchmark (built on *robosuite*) targeting lifelong
       household manipulation.  The **Goal** suite comprises four tasks
       requiring goal-conditioned reasoning and long-horizon control.
