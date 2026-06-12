基于 ManiSkill Joint-Control RLT 的两阶段训练
=================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 RLinf 中复现 **ManiSkill PegInsertionSideWideClearance-v1** 的
joint-control RLT 流程。该流程是一个
**数据采集 -> OpenPI SFT 基座 -> Stage1 RL token -> Stage2 在线 TD3**
的两阶段后训练方案。


任务与方法概览
----------------

当前 RLT ManiSkill 示例使用如下设置：

- **Environment**: ManiSkill3 ``PegInsertionSideWideClearance-v1``
- **Control mode**: ``pd_joint_delta_pos``
- **Observation**: 第三视角 RGB + wrist RGB + Panda 前 9 维 qpos
- **Action**: 10-step action chunk，每步 8 维 joint action
- **Prompt**: ``insert the peg in the hole``
- **Reward**: ``only_success`` 稀疏成功奖励

RLT 的核心思路是：**冻结大 VLA 主干，只训练小型 RL 头部**。

1. **OpenPI SFT 基座**：先得到一个能输出 joint chunk action 的 OpenPI 策略。
2. **Stage1**：训练 RL token encoder/decoder，从 VLA embedding 中学习紧凑状态表征。
3. **Stage2**：冻结 VLA 和 RL token，仅训练 direct Gaussian actor + twin-Q critic。

当前实现说明
------------

当前 RLinf 集成里，学习侧被刻意做成轻量路径：

- rollout worker 负责运行冻结的 OpenPI VLA 和冻结的 RL-token encoder
- actor worker 只训练 Stage2 的 actor 和 critic
- rollout 侧只同步 ``actor.*`` 权重

这会直接影响训练解释。如果 Stage2 行为不对，首先应检查 rollout 侧缓存特征、
learner 侧 replay，以及 actor-only weight sync 是否一致。

前置依赖
--------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 国内用户可按需使用镜像
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~

**方式一：Docker**

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

进入容器后切换环境：

.. code-block:: bash

   source switch_env openpi

**方式二：本地环境**

.. code-block:: bash

   # 国内环境可追加 --use-mirror
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

3. 下载 ManiSkill 资源
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd rlinf/envs/maniskill
   # 可选：export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets


数据格式与目录约定
------------------

Stage1 和 Stage2 都使用 ``pi05_rlt_maniskill_joint`` 这套 dataconfig。
数据集字段采用 LeRobot schema：

- ``image``: 主视角图像
- ``wrist_image``: wrist 相机图像
- ``state``: Panda qpos 前 9 维
- ``actions``: 8 维 ``pd_joint_delta_pos`` 动作
- ``task``: 语言指令

推荐把数据集准备成一个独立目录，例如：

.. code-block:: text

   /data/rlt_maniskill_joint/
   /data/rlt_maniskill_joint_videos/

如果使用绝对路径，采集脚本会直接写入该目录；如果使用相对 ``repo_id``，则需要先设置
``HF_LEROBOT_HOME``。

Step 0：准备 OpenPI SFT 基座
----------------------------

RLT Stage1/Stage2 开始前需要先准备一个 joint-control 的 OpenPI SFT checkpoint，作为后续 RL token 和在线 RL 的 base policy。

仓库内已有对应配置：

- ``examples/sft/config/rlt_maniskill_joint_pi05_sft.yaml``

启动示例：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh rlt_maniskill_joint_pi05_sft

这个阶段的关键配置包括：

- ``actor.model.model_path``: OpenPI 基座权重
- ``actor.openpi_data.repo_id``: ManiSkill joint LeRobot 数据路径
- ``actor.openpi_data.norm_stats_path``: 与数据集匹配的归一化统计
- ``actor.model.openpi.config_name``: ``pi05_rlt_maniskill_joint``

完成后会得到类似如下目录：

.. code-block:: text

   logs/<time>/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_xxx/actor

后续 Stage1 和 Stage2 的 ``model_path`` 都应指向这一类 SFT checkpoint。

Step 1：准备 joint-control 数据集
---------------------------------

公开的 RLT 流程默认你已经有一份符合 ``pi05_rlt_maniskill_joint`` 的
LeRobot 格式数据集。

至少需要包含以下字段：

- ``image``: 主视角 RGB 图像
- ``wrist_image``: wrist RGB 图像
- ``state``: Panda 前 9 维 qpos
- ``actions``: 8 维 ``pd_joint_delta_pos`` 动作
- ``task``: 指令文本

归一化统计
~~~~~~~~~~

如果这是新采集的数据集，还需要生成 ``norm_stats.json``：

.. code-block:: bash

   export HF_LEROBOT_HOME=/data
   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_rlt_maniskill_joint \
     --repo-id rlt_maniskill_joint

若你直接使用绝对路径采集，也可以把数据集移动到 ``HF_LEROBOT_HOME`` 下，再以 repo name 方式生成统计。

Step 2：运行 Stage1 RL token 训练
---------------------------------

Stage1 的目标是训练 RL token encoder/decoder，不直接更新大 VLA 主干。主配置为：

- ``examples/sft/config/rlt_stage1_maniskill_joint.yaml``
- 启动脚本：``examples/sft/train_rlt_stage1.sh``

1. 先修改配置中的关键路径
~~~~~~~~~~~~~~~~~~~~~~~~~~

至少检查以下字段：

- ``data.train_data_paths[0].dataset_path``: 指向你的 joint LeRobot 数据目录
- ``actor.openpi_data.repo_id``: 指向同一份数据
- ``actor.openpi_data.norm_stats_path``: 指向同一数据集的 ``norm_stats.json``
- ``actor.model.model_path``: 指向上一步的 OpenPI SFT 基座 checkpoint

启动命令：

.. code-block:: bash

   bash examples/sft/train_rlt_stage1.sh

若需要临时覆盖配置，也可以直接在命令后追加 Hydra 参数：

.. code-block:: bash

   bash examples/sft/train_rlt_stage1.sh rlt_stage1_maniskill_joint \
     data.train_data_paths[0].dataset_path=/data/rlt_maniskill_joint \
     actor.openpi_data.repo_id=/data/rlt_maniskill_joint \
     actor.openpi_data.norm_stats_path=/data/rlt_maniskill_joint/norm_stats.json \
     actor.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_2000/actor \
     runner.logger.logger_backends='[tensorboard]'

2. Stage1 核心配置
~~~~~~~~~~~~~~~~~~

当前默认配置里，Stage1 使用：

- ``num_action_chunks: 10``
- ``action_dim: 8``
- ``embedding_dim: 2048``
- ``vla_finetune_alpha: 0.0``，即 VLA 冻结

训练完成后，关键产物在：

.. code-block:: text

   logs/<time>/rlt_stage1_maniskill_joint/checkpoints/global_step_xxx/actor/rl_token/rl_token_model.pt

这个 ``rl_token_model.pt`` 就是 Stage2 所需的 RL token 权重。

3. 可选：评测 SFT 基座
~~~~~~~~~~~~~~~~~~~~~~

如果你想在 Stage2 之前先确认 joint SFT 路线没有跑偏，可以使用仓库中的配套
eval 配置：

- ``examples/embodiment/config/rlt_maniskill_joint_pi05_sft_eval.yaml``

评测 joint SFT checkpoint：

.. code-block:: bash

   bash examples/embodiment/eval_embodiment.sh rlt_maniskill_joint_pi05_sft_eval LIBERO \
     actor.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_xxx/actor \
     rollout.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_xxx/actor

Step 3：运行 Stage2 在线 TD3 训练
---------------------------------

Stage2 使用冻结 VLA + 冻结 RL token + 小 actor-critic 的在线 RL 流程。主配置为：

- ``examples/embodiment/config/rlt_stage2_maniskill_joint.yaml``
- 启动脚本：``examples/embodiment/run_embodiment.sh``

1. 先修改 Stage2 关键路径
~~~~~~~~~~~~~~~~~~~~~~~~~~

至少检查以下字段：

- ``actor.model.model_path``: SFT 基座 checkpoint
- ``rollout.expert_model.model_path``: expert/reference checkpoint
- ``actor.model.rlt_stage2.rl_token_path``: Stage1 导出的 ``rl_token_model.pt``
- ``actor.model.rlt_stage2.norm_stats_path``: 与 joint 数据集匹配的 ``norm_stats.json``

推荐同时把日志后端切到 ``tensorboard``，并打开 eval 视频：

.. code-block:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard"]

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

2. 启动命令
~~~~~~~~~~~

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_maniskill_joint LIBERO

如果想在不改 yaml 的情况下直接覆盖路径，可以使用：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_maniskill_joint LIBERO \
     actor.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_1000/actor \
     rollout.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_1000/actor \
     rollout.expert_model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_8000/actor \
     actor.model.rlt_stage2.rl_token_path=/path/to/rlt_stage1_maniskill_joint/checkpoints/global_step_5000/actor/rl_token/rl_token_model.pt \
     actor.model.rlt_stage2.norm_stats_path=/data/rlt_maniskill_joint/norm_stats.json \
     runner.logger.logger_backends='[tensorboard]' \
     env.eval.video_cfg.save_video=True

.. note::

   ``run_embodiment.sh`` 的第二个位置参数会写入 ``ROBOT_PLATFORM``。本配置内部
   使用的是 ``policy_setup: panda-qpos``，因此继续沿用这个 wrapper 接口即可。

3. Stage2 关键超参数
~~~~~~~~~~~~~~~~~~~~

当前配置中的重要项包括：

- ``algorithm.warmup_min_size: 1000``：replay 至少累积 1000 条 chunk transition 再训练
- ``algorithm.warmup_post_collect_updates: 30000``：actor 正式上线前先做一轮 critic warmup
- ``algorithm.train_every_transitions: 5``：每新增 5 条 replay transition 增加训练预算
- ``algorithm.max_updates_per_train_step: 1600``：限制单次 runner step 中的实际更新数
- ``actor.model.rlt_stage2.replay_subsample_stride: 0``：默认使用 chunk boundary replay
- ``actor.model.rlt_stage2.actor_noise_sigma: 0.002``：训练探索噪声
- ``actor.model.rlt_stage2.ref_action_dropout: 0.5``：避免 actor 只复制 VLA reference

4. 当前 rollout 与 replay 语义
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当前实现里最关键的行为是：

- rollout 侧会在同步到的 learner update 版本达到
  ``algorithm.warmup_post_collect_updates`` 后才让 student actor 真正接管
- 在此之前，rollout 执行的是 base VLA reference chunk
- 公开默认配置使用的是 boundary-only replay，因为
  ``replay_subsample_stride`` 为 ``0``
- stride replay 仍然存在，但属于更重的可选路径
- expert intervention 执行替换动作时，对应 step 的 replay reference chunk
  也会同步替换

5. Stage2 输出
~~~~~~~~~~~~~~

训练日志、checkpoint 和评测视频通常会落在：

.. code-block:: text

   logs/<time>-rlt_stage2_maniskill_joint/
   logs/<time>-rlt_stage2_maniskill_joint/checkpoints/
   logs/<time>-rlt_stage2_maniskill_joint/video/eval/

评测、可视化与结果展示
----------------------

1. TensorBoard
~~~~~~~~~~~~~~

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

2. 优先关注的指标
~~~~~~~~~~~~~~~~~

Stage2 中建议优先看：

- ``eval/success_once``: 最终性能指标，优先级最高
- ``env/success_once``: 在线采样中的成功率，可辅助观察，但可能受 intervention 影响
- ``train/replay_buffer/size``: replay 是否已经超过 warmup 门槛
- ``train/replay_buffer/intervention_rate``: expert 样本占比
- ``train/rlt_stage2/pending_update_budget``: learner 是否跟得上新数据

对于这个任务，**不要只看 train success 判断是否真的提升**。固定 reset ids 的
``eval/success_once`` 更适合比较 checkpoint。

3. 评测命令
~~~~~~~~~~~

如果只想单独评估某个 Stage2 checkpoint，可直接复用相同配置：

.. code-block:: bash

   bash examples/embodiment/eval_embodiment.sh rlt_stage2_maniskill_joint LIBERO \
     actor.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_1000/actor \
     rollout.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_1000/actor \
     actor.model.rlt_stage2.rl_token_path=/path/to/rlt_stage1_maniskill_joint/checkpoints/global_step_5000/actor/rl_token/rl_token_model.pt \
     runner.ckpt_path=/path/to/stage2_checkpoint.pt \
     env.eval.video_cfg.save_video=True

4. 截图/视频展示位
~~~~~~~~~~~~~~~~~~

- Stage2 ``eval/success_once`` 曲线截图
- ``logs/.../video/eval`` 中的成功/失败 episode 对比视频

下面保留一个与现有 ManiSkill 示例一致的展示块，当前可先放通用 ManiSkill 媒体，后续替换成
RLT joint 的专属结果图。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvla.png" style="width: 100%;"/>
       <p><em>Stage1 / Stage2 curve placeholder</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvlaoft.png" style="width: 100%;"/>
       <p><em>Evaluation screenshot placeholder</em></p>
     </div>
   </div>

.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src="https://github.com/RLinf/misc/raw/main/pic/embody.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

常见检查项
----------

如果流程跑不通，优先检查以下内容是否一致：

- ``actor.model.model_path`` 是否指向同一套 OpenPI joint SFT 基座
- ``actor.model.rlt_stage2.rl_token_path`` 是否来自同语义的 Stage1
- ``norm_stats_path`` 是否对应当前数据集
- ``num_action_chunks`` / ``action_dim`` 是否保持 ``10`` / ``8``
- 数据集中的 prompt 是否统一为 ``insert the peg in the hole``
- ``warmup_post_collect_updates`` 的单位是 learner 完成的 update 数，不是 runner step

如果 Stage2 出现 “train 指标上涨但 eval 不涨”，通常先排查：

- learner 是否落后于 replay：看 ``pending_update_budget``
- intervention 是否过多或几乎不触发：看 ``intervention_rate``
- base VLA / RL token / norm stats 是否错配
- eval 是否仍在使用固定且偏难的一组 reset ids
