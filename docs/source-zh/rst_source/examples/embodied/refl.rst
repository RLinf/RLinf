使用 ImageReward 微调 SD3
=========================

.. figure:: https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/figures/ImageReward.jpg
   :align: center
   :width: 80%

   ImageReward 概览，图片来自 ImageReward 作者的官方仓库。

使用 Reward Feedback Learning（ReFL）和可微的图像偏好奖励微调 Stable Diffusion 3.5。先运行纯奖励训练，再按需加入图文监督更新，比较它们对生成质量的影响。本示例复用 RLinf 的 diffusion 环境、rollout worker、FSDP actor、权重同步和训练 runner。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      Diffusion 图像生成

   .. grid-item-card:: 算法
      :text-align: center

      ReFL

   .. grid-item-card:: 任务
      :text-align: center

      文生图偏好微调

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA；下文采用四卡配置

配置文件为 ``examples/diffusion/config/sd3_refl_imagereward.yaml``。共享的 diffusion 环境和 NFT 示例见 :doc:`../video_models`。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 模型
     - 奖励
     - 训练数据
   * - 使用 LoRA 的 SD3.5 Medium
     - ImageReward 人类偏好分数
     - 文本 prompt；可选的图文对用于监督更新

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 观测
     - ``task_descriptions``：一批文本 prompt。
   * - 动作
     - 生成的 RGB 图像；每张图像对应一个完整的环境 episode。
   * - 奖励
     - ``avg`` 和 ``image_reward``：相同的标量分数，越高越好。
   * - 更新输入
     - rollout 保存的晚期去噪状态和 prompt embedding，均已 detach。

安装
----

在仓库根目录安装 diffusion 环境；通用安装要求见 :doc:`/rst_source/start/installation`：

.. code-block:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

diffusion 依赖固定了 ``image-reward==1.5`` 和 ``timm==0.6.13``。复用已有环境时，先检查这两个版本；较新的 ``timm`` 可能与 ImageReward 的依赖要求冲突。

准备模型与 Prompt
-----------------

先在 Hugging Face 接受 SD3.5 模型许可，并通过 ``hf auth login`` 登录，以下载需要授权的权重。随后下载生成模型、奖励权重、ImageReward 使用的 BERT tokenizer 和官方 prompt：

.. code-block:: bash

   export SD3_MODEL="$PWD/models/stable-diffusion-3.5-medium"
   export IR_MODEL="$PWD/models/ImageReward"
   export IR_DATA="$PWD/data/ImageReward"
   mkdir -p "$SD3_MODEL" "$IR_MODEL" "$IR_DATA"

   hf download stabilityai/stable-diffusion-3.5-medium --local-dir "$SD3_MODEL"
   hf download zai-org/ImageReward ImageReward.pt med_config.json --local-dir "$IR_MODEL"
   python -c 'from transformers import BertTokenizer; BertTokenizer.from_pretrained("bert-base-uncased")'

   for file in refl_data.json test.json; do
     curl --fail --location \
       "https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/data/$file" \
       --output "$IR_DATA/$file" || break
   done

在启动训练的 shell 中保留这些环境变量。每个节点都需要能够从 Hugging Face 缓存加载 ``bert-base-uncased``；离线运行前，应在每个节点准备好缓存。

prompt JSON 的顶层是列表，每项包含 ``text`` 或 ``prompt``。loader 去除首尾空白，按原顺序去重，并从训练集中排除 ``env.train.dataset.exclude_path`` 中的 prompt。上述固定版本的官方文件会得到 9,756 条训练 prompt 和 463 条评测 prompt。

运行纯奖励训练
--------------

先只优化奖励目标，作为后续比较监督更新影响的对照。以下四卡配置每个 runner step 生成 512 个训练样本，按 global batch size 64 更新：

.. code-block:: bash

   REFL_ARGS=(
     actor.model.model_path="$SD3_MODEL"
     rollout.model.model_path="$SD3_MODEL"
     actor.refl_reward_model.model_path="$IR_MODEL/ImageReward.pt"
     actor.refl_reward_model.med_config="$IR_MODEL/med_config.json"
     env.train.dataset.path="$IR_DATA/refl_data.json"
     env.train.dataset.exclude_path="$IR_DATA/test.json"
     env.eval.dataset.path="$IR_DATA/test.json"
     env.train.total_num_envs=4
     env.train.rollout_epoch=128
     env.eval.total_num_envs=4
     env.eval.rollout_epoch=128
     actor.global_batch_size=64
     actor.micro_batch_size=1
     actor.refl_pre_data.num_workers=0
     rollout.pipeline_stage_num=1
     env.train.reward.batch_size=1
     env.eval.reward.batch_size=1
     algorithm.refl_grad_scale=1e-3
     runner.max_epochs=200
     runner.max_steps=200
     runner.val_check_interval=10
     runner.save_interval=10
   )

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     algorithm.refl_pre_loss_weight=0.0 \
     runner.logger.experiment_name=sd3_refl_reward_only

此命令从原始模型开始训练 200 步，每 10 步评测并保存。请按 :doc:`/rst_source/start/installation` 配置机器上的 GPU；micro-batch size 1 只是起点，不保证一定满足显存限制。actor 会加载可微的 ImageReward 模型，环境中的奖励打分也会占用显存。

加入图文监督更新
----------------

将 ``algorithm.refl_pre_loss_weight`` 设为正值，会在奖励更新之外继续训练生成模型的去噪能力。这个可选的预训练目标损失称为 Lpre。请将本地图文对整理成 img2dataset 的 ``files`` 布局：

.. code-block:: text

   image_text_data/
     00000.jpg
     00000.txt
     shard_01/
       00001.png
       00001.txt

loader 会递归查找 ``.jpg``、``.jpeg``、``.png`` 和 ``.webp`` 图像。每张图像旁需要有同名的 UTF-8 ``.txt`` caption。图像会经过缩放、512 像素中心裁剪、随机水平翻转，并归一化到 [-1, 1]。默认的 ``caption_prefix`` 仅在 caption 以 ``The image shows`` 加一个空格开头时移除该前缀。

指定图像目录后，启动独立实验：

.. code-block:: bash

   export PRE_DATA=/path/to/image_text_data

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     runner.logger.experiment_name=sd3_refl_pre01

此实验保持纯奖励实验的奖励系数和 rollout 设置。将 ``algorithm.refl_pre_loss_weight`` 改为 ``0.01``，同时使用新的实验名，可比较更低权重的效果。保留 ``actor.refl_pre_data.num_workers=0``，让 actor 进程直接加载图像，避免在 accelerator 初始化后创建 DataLoader 子进程。

两种更新如何交互
~~~~~~~~~~~~~~~~

actor 从晚期 rollout 状态预测干净 latent，解码后计算奖励。VAE 和 ImageReward 的参数保持冻结，但梯度可以经过它们回传到生成模型。奖励损失为：

.. math::

   L_{\mathrm{reward}} = \lambda\,\mathbb{E}
   [\max(0, m - r(\hat{x}_0, c))],

其中 ``refl_grad_scale`` 控制 :math:`\lambda`，``refl_margin`` 控制 :math:`m`。``refl_xt_source=rollout`` 使用保存的晚期状态；``resample`` 则在采样到的晚期时间对最终 rollout latent 重新加噪。

SD3 的监督目标采用 flow matching：

.. math::

   z_t = (1-t)z_0 + t\epsilon, \qquad
   L_{\mathrm{pre}} = \mathbb{E}\left[
   \lVert v_\theta(z_t,t,c) - (\epsilon-z_0)\rVert^2\right].

对于每个 global batch，启用 Lpre 时，actor 先用 ``refl_pre_loss_weight * Lpre`` 做一次 optimizer step，再用 Lreward 做另一次 step。两次更新共享同一个 AdamW 及其动量统计，因此降低 Lpre 系数不等于按相同比例减小参数更新。系数为零时，会跳过图文数据加载和监督更新。

Lpre 学习的是提供的图文数据，而不是约束模型保持原始 SD3 的输出。数据分布不匹配或两个目标之间的干扰都可能降低 ImageReward 分数。应与纯奖励实验对比，并检查生成图片；奖励提高本身不能证明视觉质量改善。原始 SD1.4 方法见 `ImageReward 论文 <https://arxiv.org/abs/2304.05977>`_；本示例将其中的去噪监督目标适配为 SD3 的 flow matching。

验证训练路径
------------

带 Lpre 长跑前，先用三步训练检查训练、评测和 checkpoint 保存：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     env.train.rollout_epoch=16 \
     env.eval.rollout_epoch=1 \
     runner.max_epochs=3 runner.max_steps=3 \
     runner.val_check_interval=3 runner.save_interval=3 \
     runner.logger.experiment_name=sd3_refl_smoke

确认进程正常退出，并且实验目录下的 ``checkpoints/global_step_3/actor`` 包含 checkpoint 文件。如果还要在这个短程实验中触发 Lpre 数据换轮，可将 ``PRE_DATA`` 指向恰好包含 64 对图文的独立目录：四个 actor rank 每个 step 各消耗其 16 对图文，第 2、3 步便会重新创建迭代器。

在 diffusion 环境运行 CPU 单元测试：

.. code-block:: bash

   pytest tests/unit_tests/test_diffusion.py

测试使用临时生成的数据和随机初始化的小模型；外部 ImageReward 权重由小型本地模块替代，RLinf scorer 则正常运行。测试覆盖 prompt 排除、图像预处理、可微奖励、SD3 rollout 与前向的一致性、监督梯度及配置组合。它们不验证真实权重下的奖励提升，也不验证多卡 FSDP 执行；缺少可选依赖时，相应测试会跳过。

将 ``env.eval.video_cfg.save_video=true`` 可把评测样本保存到 ``env.eval.video_cfg.video_base_dir``。diffusion 环境将图像重复为视频帧，以复用录像 wrapper。

结合生成图片观察 ``eval/image_reward`` 和 ``train/actor/refl_reward``。启用 Lpre 时，再检查 ``train/actor/refl_pre_loss``、``train/actor/refl_pre_grad_norm`` 和 ``train/actor/refl_reward_grad_norm``；梯度范数之比不是 AdamW 参数更新之比。每次评测抽取 512 个样本，而测试集有 463 条 prompt，因此其中 49 条会重复；不同实验应保持相同的评测配置。共享日志流程见 :doc:`/rst_source/guides/logger`。
