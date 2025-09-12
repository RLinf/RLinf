评估 1：具身智能体场景
========================================

简介
------------
RLinf 提供了 **即开即用的评估脚本**，用于在 *训练分布内* 与 *训练分布外* 的任务中评估具身智能体的表现。  
目前支持以下两个模拟器：

- `ManiSkill3 <https://github.com/haosulab/ManiSkill>`_
- `LIBERO <https://github.com/Lifelong-Robot-Learning/LIBERO>`_

所有辅助脚本位于 ``examples/embodiment/`` 目录下；  
你只需要根据需要修改 checkpoint 路径、GPU ID 等配置即可运行。

快速开始 — ManiSkill3
------------------------

**完整启动脚本**

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
   CONFIG_NAME=YOUR_CFG_NAME      # 其中 env.eval 必须为 maniskill_ood_template

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

该脚本首先评估 12 个 **分布外（OOD）任务**，然后评估 3 个 **分布内（ID）任务**，  
所有日志保存在 ``logs/eval/<EVAL_NAME>/…/run_ppo.log``。

快速开始 — LIBERO
------------------------

**完整启动脚本**

.. code-block:: bash

   #! /bin/bash

   export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
   export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
   export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

   export MUJOCO_GL="osmesa"
   export PYOPENGL_PLATFORM="osmesa"
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

   # LIBERO 仓库路径
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

**模型路径设置**

.. code-block:: yaml

   rollout:
     model_dir: "/path/to/sft_base_model/"
   actor:
     checkpoint_load_path: "/path/to/sft_base_model/"
     model:
       ckpt_path: "/path/to/rl_ckpt.pt"
     tokenizer:
       tokenizer_model: "/path/to/sft_base_model/"

**关键 YAML 配置字段**

主 YAML + ``config/env/eval/libero_goal.yaml`` 配置文件中：

==========================  =============================================
字段名                      作用
==========================  =============================================
``simulator_type``          必须为 ``libero``
``task_suite_name``         LIBERO 任务分支名（如 ``libero_goal``）
``max_episode_steps``       每个 episode 的最大步数（默认 512）
``seed``                    环境随机种子
``num_envs``                并行评估环境数量（例如 500）
==========================  =============================================

评估结果
--------

两个评估脚本运行结束后，日志中会输出一条 **结果总结行**：

.. code-block:: javascript

   eval_metrics={
       'eval/env_info/success_once': 0.8984375,
       'eval/env_info/return': 1.0476562,
       'eval/env_info/episode_len': 80.0,
       'eval/env_info/reward': 0.0130957,
       'eval/env_info/success_at_end': 0.859375
   }

字段 ``success_once`` 表示 **成功率** （即在一次 episode 中至少完成一次任务）。  
如果启用了 TensorBoard，这些指标也会记录到 TensorBoard 中。

评估环境列表
------------------------

.. list-table:: 支持的具身智能体环境
   :header-rows: 1
   :widths: 20 80

   * - 环境
     - 简要说明
   * - ``ManiSkill3``
     - 基于 MuJoCo 的高保真模拟器，覆盖多种操作技能（如抓取、放置、推送）。  
       本次评估专注于 **Put-on-Plate** 系列任务，包含多个分布外纹理/物体组合。
   * - ``LIBERO``
     - 基于 *robosuite* 构建的大规模终身学习基准，专注于家庭任务的操控。  
       其中 **Goal** 分支包含四个任务，要求具备目标条件下的推理能力和长时间控制能力。
