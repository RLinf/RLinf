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
   CKPT_PATH=YOUR_CKPT_PATH
   CONFIG_NAME=YOUR_CFG_NAME      # env.eval must be maniskill_ood_template

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
            env.eval.init_params.id=${env_id} \
            env.eval.init_params.obj_set=${obj_set} \
            actor.model.ckpt_path=${CKPT_PATH}"
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
            env.eval.init_params.id=${env_id} \
            env.eval.init_params.obj_set=${obj_set} \
            actor.model.ckpt_path=${CKPT_PATH}"
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
   export LIBERO_REPO_PATH="/root/LIBERO"
   export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}
   export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH

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
     model_dir: "/path/to/sft_base_model/"
   actor:
     checkpoint_load_path: "/path/to/sft_base_model/"
     model:
       ckpt_path: "/path/to/rl_ckpt.pt"
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
     - A high-fidelity MuJoCo-based simulator covering diverse
       manipulation skills (grasp, place, push).  Evaluation focuses on
       **Put-on-Plate** tasks with multiple OOD texture/object splits.
   * - ``LIBERO``
     - A large-scale benchmark (built on *robosuite*) targeting lifelong
       household manipulation.  The **Goal** suite comprises four tasks
       requiring goal-conditioned reasoning and long-horizon control.
