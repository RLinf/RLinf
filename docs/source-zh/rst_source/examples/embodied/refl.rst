使用 ImageReward 微调 SD3
==========================

.. figure:: https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/figures/ImageReward.jpg
   :align: center
   :width: 80%

   ImageReward 概览，图片来自 ImageReward 作者的官方仓库。

使用 Reward Feedback Learning（ReFL）微调 Stable Diffusion 3.5 Medium，让 ImageReward 的图像偏好分数通过反向传播更新生成模型。先运行仅使用奖励的训练，再添加图文监督更新，比较两种训练方式。

概览
----

本例使用 LoRA 和 diffusion 环境。每条 prompt 生成一张图片，由 ImageReward 为图片打分。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      Diffusion 生成

   .. grid-item-card:: 算法
      :text-align: center

      ReFL

   .. grid-item-card:: 任务
      :text-align: center

      文生图偏好微调

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA；下方为四卡配置

| **操作流程：** 安装 → 准备模型和 prompt → 训练 → 检查图片与奖励。
| **前置条件：** 接受 SD3.5 模型许可，并为下方命令准备四张 NVIDIA GPU。

任务
~~~~

本例使用 ``examples/diffusion/config/sd3_refl_imagereward.yaml``。监督更新单独读取带 caption 的图片，与奖励训练使用的 prompt 数据分开准备。

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 模型
     - 奖励
     - 训练数据
   * - 使用 LoRA 的 SD3.5 Medium
     - ImageReward 人类偏好分数
     - 文本 prompt；可选的图文监督数据

观测与动作
~~~~~~~~~~

生成一张图片即完成一个环境 episode。rollout 同时保存晚期去噪状态，供 actor 计算可微奖励。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 含义
   * - 观测
     - ``task_descriptions``：一批文本 prompt。
   * - 动作
     - 生成的 RGB 图片。
   * - 奖励
     - ``avg`` 和 ``image_reward``：同一个标量分数，越高越好。
   * - 更新输入
     - rollout 保存并 detach 的晚期去噪状态和 prompt embedding。

安装
----

在仓库根目录安装并激活 diffusion 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

安装脚本选择 ``image-reward==1.5`` 及其依赖的 ``timm==0.6.13``，复用环境时也应保留这两个版本。其他安装选项见 :doc:`/rst_source/start/installation`。

下载模型
--------

在 Hugging Face 接受 SD3.5 模型许可，并运行 ``hf auth login``。下载生成模型、ImageReward 权重、tokenizer 和官方 prompt 文件：

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

在启动训练的 shell 中保留这些变量。多节点训练时，每个节点都应准备好 ``bert-base-uncased`` 的 tokenizer 缓存。

prompt 文件是 JSON 列表，每项包含 ``text`` 或 ``prompt`` 字段。加载器会清理首尾空白、去重，并通过 ``env.train.dataset.exclude_path`` 从训练集中排除评测 prompt。上述固定版本的文件得到 9,756 条训练 prompt 和 463 条评测 prompt。

运行训练
--------

先运行仅使用奖励的训练，作为添加监督更新时的对照。下方命令将 actor、env 和 rollout 放在 GPU 0–3 上。

仅使用奖励训练
~~~~~~~~~~~~~~

设置本地路径后启动 200 步训练。每个 runner step 生成 512 个训练样本，按 global batch size 64 分成八批更新：

.. code-block:: bash

   REFL_ARGS=(
     'cluster.component_placement={actor\,env\,rollout:0-3}'
     actor.model.model_path="$SD3_MODEL"
     rollout.model.model_path="$SD3_MODEL"
     actor.model.sd3.lora_path=null
     actor.refl_reward_model.type=image_reward
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
     actor.seed=42
     env.train.seed=42
     env.eval.seed=0
     runner.max_epochs=200
     runner.max_steps=200
     runner.val_check_interval=10
     runner.save_interval=10
     runner.resume_dir=null
     runner.ckpt_path=null
   )

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     algorithm.refl_pre_loss_weight=0.0 \
     runner.logger.experiment_name=sd3_refl_reward_only

训练每 10 步评测并保存一次 checkpoint。首次运行可以覆盖为 ``runner.max_epochs=2``、``runner.max_steps=2``、``runner.val_check_interval=2`` 和 ``runner.save_interval=-1``，先检查短流程。micro-batch size 1 是起始设置；actor 的可微打分和环境打分都会占用 GPU 显存。placement 的配置方式见 :doc:`/rst_source/concepts/placement`。

``actor.refl_reward_model.type`` 选择 actor 使用的 scorer，默认值为 ``image_reward``。环境通过 ``env.*.reward.model`` 为生成结果打分，本例设为 ``visual.image_reward``。

配置奖励更新
~~~~~~~~~~~~

actor 从晚期去噪状态出发，一步预测干净 latent，再解码成图片。VAE 和 ImageReward 的参数保持冻结，但梯度会经过它们传回生成模型。奖励 loss 为：

.. math::

   L_{\mathrm{reward}} = \lambda\,\mathbb{E}
   [\max(0, m - r(\hat{x}_0, c))].

``refl_grad_scale`` 对应 :math:`\lambda`，``refl_margin`` 对应 :math:`m`；分数达到 :math:`m` 的样本，其奖励梯度为零。以下配置位于 ``algorithm`` 下：

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - 配置
     - 默认值
     - 作用
   * - ``refl_margin``
     - ``2.0``
     - hinge loss 的奖励阈值。
   * - ``refl_grad_scale``
     - ``1e-3``
     - 奖励 loss 的缩放系数。
   * - ``refl_xt_source``
     - ``rollout``
     - 选择保存的晚期状态；``resample`` 则对最终 latent 在采样的晚期时间重新加噪。
   * - ``refl_pre_loss_weight``
     - ``0.0``
     - 大于零时，启用下方的监督更新。

`ImageReward 论文 <https://arxiv.org/abs/2304.05977>`_ 使用 SD1.4 介绍 ReFL，本例将奖励反馈用于 SD3 flow matching。

添加图文监督更新
~~~~~~~~~~~~~~~~

要在奖励更新之外继续进行去噪训练，先准备 img2dataset ``files`` 格式的本地图文对。这个可选监督目标称为 Lpre：

.. code-block:: text

   image_text_data/
     00000.jpg
     00000.txt
     shard_01/
       00001.png
       00001.txt

加载器递归查找 ``.jpg``、``.jpeg``、``.png`` 和 ``.webp`` 图片，每张图片对应同名的 UTF-8 ``.txt`` caption。图片会缩放并中心裁剪到 512 像素，随机水平翻转，再归一化到 ``[-1, 1]``。默认的 ``caption_prefix`` 会移除 caption 开头的 ``The image shows`` 及其后的空格。可以使用自己的图文对，也可以将 `PD3M <https://huggingface.co/datasets/Spawning/PD3M>`_ 子集整理成该格式。

设置图文目录，并复用上方配置：

.. code-block:: bash

   export PRE_DATA=/path/to/image_text_data

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     runner.logger.experiment_name=sd3_refl_pre01

每个 global batch 先对 ``refl_pre_loss_weight * Lpre`` 执行一次 optimizer step，再对奖励 loss 执行一次。Lpre 使用 SD3 的 flow-matching 去噪目标；权重为零时，跳过图片加载和监督更新。

两次更新共享 AdamW 的动量统计，因此权重 ``0.01`` 不表示监督参数更新量是权重 ``1.0`` 时的 1%。示例中的 ``0.1`` 用于对照实验，不代表已确定的最优值。比较权重时，固定数据集、初始模型、学习率、seed、batch 和训练长度。按上方配置，Lpre 每个 runner step 会额外执行八次 optimizer step，也应记录增加的计算开销。

可视化与结果
------------

结合生成图片观察 ``eval/image_reward`` 和 ``train/actor/refl_reward``。启用 Lpre 后，再检查 ``train/actor/refl_pre_loss``、``train/actor/refl_pre_grad_norm`` 和 ``train/actor/refl_reward_grad_norm``。Lpre 记录未加权的 loss，奖励 loss 则包含 ``refl_grad_scale``。ImageReward 分数上升本身并不等于视觉质量改善。

每次评测用 463 条 prompt 生成 512 张图片，因此有 49 条 prompt 会重复。对照实验应保持相同的评测设置。将 ``env.eval.video_cfg.save_video=true`` 可把生成图片保存为重复帧视频，输出目录为 ``env.eval.video_cfg.video_base_dir``。

日志位置见 :doc:`/rst_source/guides/logger`，共享指标见 :doc:`/rst_source/reference/metrics`。如需更换 scorer 或适配其他模型，参见 :ref:`ReFL 接口说明 <refl-model-interface>`。
