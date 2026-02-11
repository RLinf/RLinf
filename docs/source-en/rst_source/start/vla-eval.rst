Evaluation Tutorial 1: Embodied VLA
====================================

Introduction
------------
RLinf provides **out-of-the-box evaluation scripts** to evaluate the performance of embodied agents in both *in-distribution* and *out-of-distribution* tasks.
List of currently supported evaluation environments:

:doc:`Behavior <../examples/embodied/behavior>`

:doc:`Calvin <../examples/embodied/calvin>`

:doc:`Isaaclab <../examples/embodied/isaaclab>`

:doc:`Libero <../examples/embodied/libero>`

:doc:`ManiSkill <../examples/embodied/maniskill>`

:doc:`MetaWorld <../examples/embodied/metaworld>`

:doc:`RoboCasa <../examples/embodied/robocasa>`

All startup scripts for evaluation are located in the ``examples/embodiment/`` directory.

Quick Start
------------------------

**Evaluation Launch Command**

.. code-block:: bash

  # run eval：
  bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft_eval

**Key YAML Configuration Fields**

Any YAML file can be used for evaluation with the ``eval_embodiment.sh`` script, provided it includes the relevant ``env.eval`` configuration. Taking ``examples/embodiment/config/libero_10_grpo_openvlaoft_eval.yaml`` as an example here, you can modify the following in the configuration file as needed:

1. **Adjust model path** : Modify the following two parameters to load the model to be evaluated:

  1. ``rollout.model.model_path``

  2. ``runner.ckpt_path`` (Optional) – ``.pt`` format file path, set this parameter if you want to evaluate a specific checkpoint.

.. Note::
  
   If you need to convert checkpoint files from ``.distcp`` format to ``.pt`` format, please refer to the :doc:`Checkpoint Convertor <../tutorials/advance/convertor>` documentation for detailed instructions.

2. **Control environment random seed**: You can adjust ``env.seed`` to change the environment's random function for result reproducibility, etc.

.. Note::
  
   When multiple workers launch environments, the seeds in different workers have a fixed offset: ``seed = seed + self._rank * self.stage_num + stage_id``.

3. **Adjust evaluation epochs**: We can adjust ``algorithm.eval_rollout_epoch`` to control the number of evaluation epochs. Note that we assume each epoch should complete the evaluation of the entire test set. Furthermore, since the random seeds are identical for each evaluation, the final **evaluation result** is equivalent to the average result of the Policy evaluated over multiple rounds on the same test set.

4. **Adjust the number of evaluation environments per epoch**: To fully evaluate the entire test set (for example, Libero-10 has 500 initial states while Libero-90 has 4500 initial states), we can adjust the number of loaded environments in the following two ways:

  1. The first method is to increase ``env.eval.total_num_envs`` to control the number of environments loaded in parallel (distributed evenly across all workers). However, this can easily lead to OOM (Out Of Memory) issues in resource-constrained settings, for instance, if you only have a single 40GB GPU;

  2. Therefore, we offer an alternative: enable ``env.eval.auto_reset=True`` and then adjust ``max_steps_per_rollout_epoch`` to be **N** times the value of ``max_episode_steps``. In this case, the total number of environments in each ``eval_rollout_epoch`` evaluated will be ``N * env.eval.total_num_envs``;

5. **Adjust max interaction steps per trajectory**: You can adjust ``env.eval.max_episode_steps`` to control the maximum number of interaction steps in a single trajectory.

6. **Record environment video**: You can set ``env.eval.video_cfg.save_video=True`` to record videos of the environment during evaluation.

7. **Control evaluation sampling method**: You can adjust ``algorithm.sampling_params`` to control the sampling method during rollout evaluation.

8. **Eval is set True**: You can set the value ``cfg.runner.only_eval=True``, and we also set it automatically in ``eval_embodied_agent.py`` which is called by ``eval_embodimen.sh``.

**Evaluation Launch Script**

.. code-block:: bash

   #! /bin/bash

  export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
  export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
  export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

  export MUJOCO_GL="osmesa"
  export PYOPENGL_PLATFORM="osmesa"
  export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

  # Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
  # Only required when running the behavior experiment.
  export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
  export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
  export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
  export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
  export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
  # Base path to Isaac Sim, only required when running the behavior experiment.
  export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
  export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
  export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

  export HYDRA_FULL_ERROR=1


  if [ -z "$1" ]; then
      CONFIG_NAME="maniskill_ppo_openvlaoft"
  else
      CONFIG_NAME=$1
  fi

  LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
  MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
  mkdir -p "${LOG_DIR}"
  CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
  echo ${CMD}
  ${CMD} 2>&1 | tee ${MEGA_LOG_FILE}


After finish the evaluation, **Evaluation Results** will be output to the terminal and log files:

.. code-block:: javascript

  [INFO 04:00:43 RLinf] {
    'eval/success_once': array(0.11328125, dtype=float32), 
    'eval/return': array(0.43945312, dtype=float32), 
    'eval/episode_len': array(512., dtype=float32), 
    'eval/reward': array(0.00085831, dtype=float32), 
    'eval/success_at_end': array(0.08789062, dtype=float32)
  }

The field ``success_once`` represents the **success rate** (i.e., completing the task at least once within a single episode trajectory).
If TensorBoard is enabled, these metrics will also be recorded in TensorBoard (TensorBoard is enabled by default).


Quick Start — ManiSkill3 OOD
--------------------------------

.. note::
  
   Currently, only ManiSkill is supported for OOD test.

**Launch Method**

Modify variables such as ``EVAL_NAME``, ``CKPT_PATH``, and ``CONFIG_NAME`` (which can be set to ``maniskill_ppo_openvlaoft_quickstart`` for a quick test) in ``examples/embodiment/eval_mani_ood.sh``.
Then, execute the following command in the terminal to start the evaluation.

.. code-block:: bash

  bash examples/embodiment/eval_mani_ood.sh

**Full Launch Script**

.. code-block:: bash

  #! /bin/bash
  export HF_ENDPOINT=https://hf-mirror.com

  export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
  export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
  export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

  export HYDRA_FULL_ERROR=1

  EVAL_NAME=YOUR_EVAL_NAME
  CKPT_PATH=YOUR_CKPT_PATH           # Optional: .pt file or None, if None, will use the checkpoint in rollout.model.model_path
  CONFIG_NAME=YOUR_CFG_NAME          # env.eval must be maniskill_ood_template
  TOTAL_NUM_ENVS=YOUR_TOTAL_NUM_ENVS # total number of evaluation environments
  EVAL_ROLLOUT_EPOCH=YOUR_EVAL_ROLLOUT_EPOCH # eval rollout epoch, total_trajectory_num = eval_rollout_epoch * total_num_envs

  for env_id in \
      "PutOnPlateInScene25VisionImage-v1" "PutOnPlateInScene25VisionTexture03-v1" "PutOnPlateInScene25VisionTexture05-v1" \
      "PutOnPlateInScene25VisionWhole03-v1"  "PutOnPlateInScene25VisionWhole05-v1" \
      "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25Plate-v1" "PutOnPlateInScene25Instruct-v1" \
      "PutOnPlateInScene25MultiCarrot-v1" "PutOnPlateInScene25MultiPlate-v1" \
      "PutOnPlateInScene25Position-v1" "PutOnPlateInScene25EEPose-v1" "PutOnPlateInScene25PositionChangeTo-v1" ; \

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
          runner.ckpt_path=${CKPT_PATH}"

      echo ${CMD} > ${MEGA_LOG_FILE}
      ${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
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
          env.eval.total_num_envs=${TOTAL_NUM_ENVS} \
          env.eval.init_params.id=${env_id} \
          env.eval.init_params.obj_set=${obj_set} \
          runner.ckpt_path=${CKPT_PATH}"
      echo ${CMD}  > ${MEGA_LOG_FILE}
      ${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
  done

This script first evaluates 13 **Out-Of-Distribution (OOD) tasks**, and then evaluates 3 **In-Distribution (ID) tasks**.
All logs are saved in ``logs/eval/<EVAL_NAME>/…/run_ppo.log``.


