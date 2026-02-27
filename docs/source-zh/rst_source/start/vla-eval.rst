评估教程1：具身VLA
========================================

简介
------------
RLinf 提供了 **即开即用的评估脚本**，用于在 *训练分布内* 与 *训练分布外* 的任务中评估具身智能体的表现。  
目前支持的评估环境列表：

1. :doc:`Behavior <../examples/embodied/behavior>`

2. :doc:`Calvin <../examples/embodied/calvin>`

3. :doc:`Isaaclab <../examples/embodied/isaaclab>`

4. :doc:`Libero <../examples/embodied/libero>`

5. :doc:`ManiSkill <../examples/embodied/maniskill>`

6. :doc:`MetaWorld <../examples/embodied/metaworld>`

7. :doc:`RoboCasa <../examples/embodied/robocasa>`

所有关于eval的启动脚本均位于 ``examples/embodiment/`` 目录下；

快速开始
------------------------

**Eval启动命令**

.. code-block:: bash

  # 运行 eval 的实例：
  bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft_eval

**关键 YAML 配置字段**

所有yaml只要添加了 ``env.eval`` 的相关配置，均可使用 ``eval_embodiment.sh`` 脚本进行评测，在这里我们以 ``examples/embodiment/config/libero_10_grpo_openvlaoft_eval.yaml`` 为例，您可以按需修改配置文件中的：

1. 调整模型路径，同时修改以下两个参数以加载待测评的模型；

  1. ``rollout.model.model_path``

  2. ``runner.ckpt_path`` （Optional）- ``.pt`` 格式文件路径，如果需要评测指定checkpoint，可以设置该参数。

.. note::

  如果需要将 checkpoint 文件从 ``.distcp`` 格式转换为 ``.pt`` 格式，请参考 :doc:`Checkpoint 转换 <../tutorials/advance/convertor>` 文档中的详细说明。

2. 控制环境的随机种子：我们可以调整 ``env.seed`` 来调整环境的随机函数的变化，以便复现结果等；

.. note::

  多个worker启动环境时，不同worker中的环境的 ``seed`` 都有固定的偏移 ``seed = seed + self._rank * self.stage_num + stage_id``；

3. 调整测评的轮数：我们可以调整 ``algorithm.eval_rollout_epoch`` 以控制测评的轮数。注意，我们认为每轮应该测评完整个测试集，并且由于每次测评的种子都是相同的，所以，最终的 **测评结果** 等价于 Policy 在相同测试集上测评多轮取平均的结果；

4. 调整每一轮测评环境数：为了完整评估整个测试集（例如，libero-10 有500个 states 而 libero-90 有4500个 states）。我们可以通过如下两种方式调整加载环境的数量：

  1. 第一种是调大 ``env.eval.total_num_envs`` 以控制同时并行加载的环境数（均匀分布在所有workers上），但是这在资源受限的环境下容易 OOM， 比如您只有一张40g的显卡；

  2. 所以我们有另一种，也就是打开 ``env.eval.auto_reset=True`` 然后，调整 ``max_steps_per_rollout_epoch`` 为 ``max_episode_steps`` 的N倍，那么每一轮 ``eval_rollout_epoch`` 中的总环境数量将会是 ``N*env.eval.total_num_envs`` 个；

5. 调整单条轨迹交互步数：我们可以调整 ``env.eval.max_episode_steps`` 以控制单条轨迹的交互步数；

6. 录制环境视频：我们可以打开 ``env.eval.video_cfg.save_video=True`` 以录制测评时环境的视频；

7. 控制测评的采样方式：我们可以调整 ``algorithm.sampling_params`` 以控制测评时 rollout 时的采样方式；

8. 开启eval：在测评时，我们会开启 ``cfg.runner.only_eval=True``，在 ``eval_embodimen.sh`` 调用的 ``eval_embodied_agent.py`` 也会自动修改该值为 ``True``。

**Eval启动脚本**

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


测评运行结束后在终端以及在日志文件中会输出  **测评结果**：

.. code-block:: javascript

  [INFO 04:00:43 RLinf] {
    'eval/success_once': array(0.11328125, dtype=float32), 
    'eval/return': array(0.43945312, dtype=float32), 
    'eval/episode_len': array(512., dtype=float32), 
    'eval/reward': array(0.00085831, dtype=float32), 
    'eval/success_at_end': array(0.08789062, dtype=float32)
  }

字段 ``success_once`` 表示 **成功率** （即在一条 episode 轨迹中至少完成一次任务）。  
如果启用了 TensorBoard，这些指标也会记录到 TensorBoard 中（ TensorBoard 默认开启）。


快速开始 — ManiSkill3 OOD
---------------------------
**OOD Eval启动命令**

.. note::
  
  目前只支持 ManiSkill

启动方式如下： 修改 ``examples/embodiment/eval_mani_ood.sh`` 中的 EVAL_NAME, CKPT_PATH, CONFIG_NAME(可改为maniskill_ppo_openvlaoft_quickstart进行一个快速测试)等
然后在终端执行如下命令启动测评。

.. code-block:: bash

  bash examples/embodiment/eval_mani_ood.sh

**完整启动脚本**

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

该脚本首先评估 13 个 **分布外（OOD）任务**，然后评估 3 个 **分布内（ID）任务**，  
所有13+3个任务的测评结果日志将会依次保存在 ``logs/eval/<EVAL_NAME>/…/run_ppo.log``。
