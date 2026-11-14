NFT 扩散模型微调
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   使用 Diffusion-NFT 训练的 SD3 与 Wan2.2 生成模型。

使用 Diffusion-NFT 微调 SD3 或 Wan2.2，改善图像和视频中的文字渲染。训练对同一 prompt 的候选做 OCR 评分，再用相对奖励更新模型。完成 :doc:`共用准备步骤 <video_models>` 后，选择下方配置启动训练，并检查 reward 和生成样本。

概览
----

按模型和输出类型选择配置。目标函数与推导见 `DiffusionNFT 论文 <https://arxiv.org/abs/2509.16117>`_ 和 `官方代码 <https://github.com/NVlabs/DiffusionNFT>`_。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: 算法
      :text-align: center

      Diffusion-NFT

   .. grid-item-card:: 输出
      :text-align: center

      图像 / 视频

   .. grid-item-card:: 奖励
      :text-align: center

      OCR / 视频 OCR

| **操作流程：** 选择配置 → 启动训练 → 检查 OCR reward 和生成样本。
| **前置条件：** :doc:`共用安装、模型与数据集准备 <video_models>`。

任务
~~~~

配置覆盖图像 OCR 与保留逐帧奖励的视频 OCR。

.. list-table::
   :header-rows: 1
   :widths: 22 34 18 26

   * - 模型
     - 配置
     - 输出
     - Reward 粒度
   * - SD3
     - ``sd3_nft_ocr``
     - 图像
     - 每张图像一个 OCR 分数
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_nft_ocr``
     - 单帧图像
     - 每张图像一个 OCR 分数
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_nft_video_ocr``
     - 视频
     - 对视频帧计算 OCR 分数

观测与动作
~~~~~~~~~~

每个 episode 对当前 prompt 生成的一张图像或一段视频打分。

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 数据
     - 含义
   * - 观测
     - 含有引号内目标文字的文本 prompt。
   * - 动作
     - 生成的图像或视频。
   * - 奖励
     - 图像 OCR 相似度或视频逐帧 OCR 分数。

安装与资源准备
--------------

按共用步骤完成 :ref:`安装 <diffusion-model-setup>`、:ref:`下载模型 <diffusion-model-checkpoints>` 和 :ref:`下载 OCR 数据集 <diffusion-ocr-dataset>`。Wan NFT 配置使用 GPU OCR，请选择共用安装步骤中的 ``--paddle-gpu`` 命令。

运行训练
--------

在 ``examples/diffusion/config/`` 下打开所选配置对应的 YAML，例如 SD3 使用 ``examples/diffusion/config/sd3_nft_ocr.yaml``。将 ``actor.model.model_path``、``env.train.dataset.path`` 和 ``env.eval.dataset.path`` 设为下载目录，并根据实际硬件检查 ``cluster.num_nodes``、``cluster.component_placement`` 及 batch 设置。

检查 YAML 后，在仓库根目录传入对应配置名启动训练：

.. code:: bash

   bash examples/diffusion/run_diffusion.sh sd3_nft_ocr

.. code:: bash

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_nft_video_ocr

启动脚本生成候选、计算 OCR 奖励并更新 LoRA 参数，日志写入 ``logs/<timestamp>-<config>/``。

将 ``runner.resume_dir`` 设为 ``checkpoints/global_step_N`` 目录可恢复 NFT 训练，包括 off-policy 的 EMA rollout 权重。如果旧 checkpoint 缺少 EMA 权重，actor 会记录告警，并用恢复后的 actor 权重初始化 EMA，此时无法精确续训。

配置 NFT 更新
--------------

当前配置将组内 reward 归一化与扩散 velocity space 中的 NFT 更新结合起来：

1. 每个 prompt 生成 ``group_size`` 个候选并计算相对 advantage。图像配置在每个 prompt group 内使用 GRPO 归一化；视频配置则以整段视频为组，对帧/chunk reward 进行归一化。
2. 采样一个去噪 step，并重建当前噪声状态（``nft_xcur_source: resample``）。rollout/reference 模型提供旧 velocity :math:`v_{old}`，可训练模型预测 :math:`v_\theta`。
3. 构造 :math:`\Delta v = v_\theta - v_{old}`，以及正负候选 :math:`v_{\pm} = v_{old} \pm \beta\Delta v`。
4. 将两个候选映射到目标空间。当前配置使用 ``nft_target_space: x0``，在 clean sample 空间中计算加权平方误差 energy :math:`E_{+}` 和 :math:`E_{-}`。
5. 将相对 advantage 映射到 ``[0, 1]`` 后计算 NFT objective。较好的候选更强调 :math:`E_{+}`，较差的候选更强调 :math:`E_{-}`；``nft_clip_ratio`` 可限制 velocity update 相对 :math:`v_{old}` 的幅度。

``nft_tau`` 控制 rollout/reference 权重。标量 ``1.0`` 表示 on-policy；列表 ``[start_tau, end_tau, start_step, end_step]`` 表示线性退火的 EMA 更新。例如 ``[1.0, 0.01, 0, 70]`` 会在前 70 次更新中，从当前策略逐渐过渡到缓慢更新的 reference model。

.. list-table:: 关键 Diffusion-NFT 参数
   :header-rows: 1
   :widths: 28 24 48

   * - 参数
     - 当前配置
     - 作用
   * - ``algorithm.group_size``
     - 图像 ``24``，视频 ``8``
     - 每个 prompt 参与相对比较的候选数量。
   * - ``nft_beta``
     - ``0.1``
     - 正负 velocity perturbation 的缩放系数。
   * - ``nft_target_space``
     - ``x0``
     - 计算候选 prediction error 的目标空间。
   * - ``nft_weight_mode``
     - ``adaptive``
     - 使用停止梯度的平均绝对误差对 energy 进行自适应归一化。
   * - ``nft_clip_ratio``
     - Wan2.2 使用 ``0.1``
     - 限制 velocity update 相对 :math:`v_{old}` 的幅度。
   * - ``nft_tau``
     - 按模型设置退火区间
     - 控制 rollout/reference 权重跟随 actor 的速度。
   * - ``actor.model.*.use_lora``
     - ``True``
     - 只训练参数高效 adapter，而不是全部模型权重。

预期实验结果
------------

.. figure:: https://github.com/user-attachments/assets/6161b286-8df9-41c3-945e-cf30ffd9f185
   :align: center
   :width: 92%

   SD3 图像 OCR、Wan2.2 图像 OCR 和 Wan2.2 视频 OCR 训练的参考
   ``env/avg`` reward 曲线。

在上面的参考实验中，三个配置都从较低的初始 OCR reward 上升到约 ``0.8``。Wan2.2 图像和视频实验上升更快，SD3 随后达到接近的区间。该图用于说明预期训练趋势，并不保证完全相同的最终分数；随机种子、prompt、初始 checkpoint、GPU 数量、依赖版本和评测设置都会影响收敛速度与最终 reward。

健康的训练通常应表现为 ``env/avg`` 持续提高，同时 NFT update 相关统计保持在合理范围。除了 scalar reward，也应检查实际生成样本：OCR 可以衡量文字匹配，但无法单独反映视觉质量和视频时序一致性。

.. list-table:: 建议关注的指标
   :header-rows: 1
   :widths: 28 72

   * - 指标
     - 含义
   * - ``env/avg``
     - ``reward.key`` 选择的 reward，也是当前示例的主要任务指标。
   * - ``actor/nft_loss``
     - 实际优化的 NFT objective，应结合 reward 判断，而不是直接作为质量分数。
   * - ``actor/nft_tau``
     - 根据退火区间计算出的当前 rollout/reference EMA 系数。
   * - ``actor/delta_v_norm``
     - 相对 rollout/reference prediction 的 velocity update 大小。
   * - ``actor/clip_frac`` / ``actor/clip_loss_frac``
     - 受 NFT clipping 影响的 update 比例；长期饱和通常表示更新过强。
   * - ``actor/E_pos_mean`` / ``actor/E_neg_mean``
     - 构造 NFT objective 使用的正负候选 energy。

使用 TensorBoard 查看日志。如果在 ``env.*.video_cfg`` 中启用媒体保存，生成样本会写入 ``video_base_dir``。

.. code:: bash

   tensorboard --host 0.0.0.0 --logdir logs/

Scalar 定义与 logger 目录结构见 :doc:`训练指标 </rst_source/reference/metrics>`。
