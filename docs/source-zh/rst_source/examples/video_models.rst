视频生成模型的强化学习
======================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   在 RLinf 中使用 Diffusion-NFT 训练图像与视频生成模型。

RLinf 集成了 Diffusion-NFT，用于对 SD3 和 Wan2.2 进行强化学习微调。该流程将
生成的图像或视频视为 action，在 RLinf 的 ``diffusion`` 环境中完成 reward
打分，再使用 reward 更新扩散模型。当前示例以文字渲染为目标，支持图像级 OCR
reward 和面向视频帧的 Video OCR reward。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      ``diffusion``

   .. grid-item-card:: 算法
      :text-align: center

      Diffusion-NFT

   .. grid-item-card:: 模型
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: 奖励
      :text-align: center

      OCR / Video OCR

| **你将完成：** 安装依赖 → 下载模型与 prompt 数据 → 选择配置 → 启动 ``run_diffusion.sh`` → 观察 reward 和 NFT 指标。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · SD3 或 Wan2.2 checkpoint · OCR prompt 数据集。

支持的任务
~~~~~~~~~~

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

Diffusion 环境如何工作
----------------------

Diffusion 环境将媒体生成任务适配到 RLinf 的环境接口。它是一个单步环境：
``reset()`` 采样 prompt，rollout worker 生成图像或视频，``step()`` 对媒体打分并
结束 episode。

1. **构造 prompt group。** 数据集采样 prompt，并按照
   ``algorithm.group_size`` 复制。同一 prompt 生成的候选样本构成一个比较组。
2. **生成候选媒体。** SD3 或 Wan2.2 根据 prompt 生成图像或视频，同时保存 NFT
   更新需要的去噪状态。
3. **计算 reward。** 配置指定的 reward backend 返回一组分数，
   ``reward.key`` 选择真正用于 RL 的标量或帧序列；当前 OCR 配置使用 ``avg``。
4. **结束 episode。** 环境将分数作为 reward 返回并终止单步 episode。对于视频
   reward，可使用 ``reward.frame_interval`` 将逐帧分数组合为环境 step。

.. list-table:: Diffusion 环境的数据接口
   :header-rows: 1
   :widths: 22 34 44

   * - 阶段
     - 数据
     - 含义
   * - Observation
     - ``task_descriptions``
     - 从 ``env.*.dataset`` 采样的文字 prompt。
   * - Action
     - 图像或视频 tensor
     - 扩散策略返回的生成媒体。
   * - Reward
     - ``scores[reward.key]``
     - 图像级分数，或逐帧/逐 chunk 分数序列。
   * - Episode 指标
     - ``return``、``avg``、backend keys
     - RLinf 记录的训练与评测指标。

该环境并不与 OCR 强绑定。数据集和 reward 实现分别通过 ``dataset.type`` 与
``reward.model`` 从 ``rlinf.data.datasets.diffusion`` 和
``rlinf.envs.sim.diffusion`` 动态加载，因此其他媒体评分器也可以复用同一个单步接口。

Reward 打分
-----------

图像 OCR Reward
~~~~~~~~~~~~~~~

OCR backend 从 prompt 中提取引号内的目标文字，对生成图像运行 PaddleOCR，移除
空格后使用 Levenshtein distance 比较识别结果与目标文字。归一化 reward 为

.. math::

   r_{\mathrm{OCR}} = 1 -
   \frac{\min\left(d(\hat{y}, y), |y|\right)}{|y|},

其中 :math:`y` 是目标文字，:math:`\hat{y}` 是 OCR 识别结果。分数接近 ``1``
表示文字渲染正确；``0`` 表示编辑距离至少达到目标字符串长度。目标文字作为完整
子串出现时也会得到 ``1``。

视频 OCR Reward
~~~~~~~~~~~~~~~

Wan2.2 视频训练会对每一帧运行相同的 OCR scorer。Backend 可以输出逐帧 reward，
也可以输出视频平均 reward。当前 ``wan22_ti2v_5b_nft_video_ocr`` 配置使用逐帧
reward，通过 ``frame_interval: 4`` 组合相邻帧，并使用 ``grpo_video`` 与
``advantage_mode: video`` 计算 advantage。这样既保留帧/chunk 级 credit，又能在
同一 prompt 的候选视频之间进行统一归一化。

接入其他 reward 时，应明确分数范围与方向：更高的 reward 必须代表更好的样本；
同一 prompt group 内也需要有足够的分数差异，才能形成有效的相对 advantage。

Diffusion-NFT 更新
------------------

当前配置将组内 reward 归一化与扩散 velocity space 中的 NFT 更新结合起来：

1. 每个 prompt 生成 ``group_size`` 个候选并计算相对 advantage。图像配置在每个
   prompt group 内使用 GRPO 归一化；视频配置则以整段视频为组，对帧/chunk reward
   进行归一化。
2. 采样一个去噪 step，并重建当前噪声状态（``nft_xcur_source: resample``）。
   rollout/reference 模型提供旧 velocity :math:`v_{old}`，可训练模型预测
   :math:`v_\theta`。
3. 构造 :math:`\Delta v = v_\theta - v_{old}`，以及正负候选
   :math:`v_{\pm} = v_{old} \pm \beta\Delta v`。
4. 将两个候选映射到目标空间。当前配置使用 ``nft_target_space: x0``，在 clean
   sample 空间中计算加权平方误差 energy :math:`E_{+}` 和 :math:`E_{-}`。
5. 将相对 advantage 映射到 ``[0, 1]`` 后计算 NFT objective。较好的候选更强调
   :math:`E_{+}`，较差的候选更强调 :math:`E_{-}`；``nft_clip_ratio`` 可限制
   velocity update 相对 :math:`v_{old}` 的幅度。

``nft_tau`` 控制 rollout/reference 权重。标量 ``1.0`` 表示 on-policy；列表
``[start_tau, end_tau, start_step, end_step]`` 表示线性退火的 EMA 更新。例如
``[1.0, 0.01, 0, 70]`` 会在前 70 次更新中，从当前策略逐渐过渡到缓慢更新的
reference model。

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

安装
----

.. include:: embodied/_setup_common.rst

**自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 --use-mirror 到下面的 install.sh 命令
   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

如果要使用自定义虚拟环境目录，可以传入 ``--venv <dir>``：

.. code:: bash

   bash requirements/install.sh embodied --model diffusion --venv /path/to/venv
   source /path/to/venv/bin/activate

该命令会创建 Python 3.10 环境，并安装 SD3、Wan2.2 和 OCR reward 所需的
Diffusers、PEFT、Transformers、PaddleOCR、PaddlePaddle 等依赖。

.. warning::

   当前 diffusion 配置使用 ``/path/to/...`` 占位路径。启动前，请按配置文件旁边
   的注释替换模型与数据集路径，或通过命令行覆盖
   ``actor.model.model_path`` 和 ``env.*.dataset.path``。

.. note::

   Wan2.2 需要支持 ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` 的 Diffusers 版本。
   请优先使用上面的安装命令，不要直接复用已经为其他模型固定旧版 Diffusers 的
   具身环境。

下载模型
--------

训练前，请下载对应的 Diffusers checkpoint，并将 ``actor.model.model_path``
设置为本地模型目录。

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - 模型
     - Hugging Face Repo
     - 本地路径示例
   * - Stable Diffusion 3.5 Medium
     - `stabilityai/stable-diffusion-3.5-medium <https://huggingface.co/stabilityai/stable-diffusion-3.5-medium>`__
     - ``/path/to/stable-diffusion-3.5-medium``
   * - Wan2.2 TI2V 5B
     - `Wan-AI/Wan2.2-TI2V-5B-Diffusers <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>`__
     - ``/path/to/Wan2.2-TI2V-5B-Diffusers``

.. code:: bash

   huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
      --local-dir /path/to/stable-diffusion-3.5-medium

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
      --local-dir /path/to/Wan2.2-TI2V-5B-Diffusers

Stable Diffusion 3.5 Medium 需要先在 Hugging Face 接受模型访问条款。

下载数据集
----------

示例使用
`NVlabs/DiffusionNFT <https://github.com/NVlabs/DiffusionNFT/tree/main/dataset/ocr>`__
中的 OCR prompt 数据集。目录中必须包含 ``train.txt`` 与 ``test.txt``；每一行是一个
prompt，``env.*.dataset.split`` 用于选择对应文件。

.. code:: bash

   mkdir -p /path/to/dataset/ocr
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt \
      -o /path/to/dataset/ocr/train.txt
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt \
      -o /path/to/dataset/ocr/test.txt

运行训练
--------

将 Diffusion-NFT 配置名传给 ``run_diffusion.sh``，并覆盖本地模型与数据集路径。

.. code:: bash

   bash examples/diffusion/run_diffusion.sh sd3_nft_ocr \
      actor.model.model_path=/path/to/stable-diffusion-3.5-medium \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

.. code:: bash

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_nft_video_ocr \
      actor.model.model_path=/path/to/Wan2.2-TI2V-5B-Diffusers \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

更多参数位于 ``examples/diffusion/config/*.yaml``。

预期实验结果
------------

.. figure:: https://github.com/user-attachments/assets/6161b286-8df9-41c3-945e-cf30ffd9f185
   :align: center
   :width: 92%

   SD3 图像 OCR、Wan2.2 图像 OCR 和 Wan2.2 视频 OCR 训练的参考
   ``env/avg`` reward 曲线。

在上面的参考实验中，三个配置都从较低的初始 OCR reward 上升到约 ``0.8``。
Wan2.2 图像和视频实验上升更快，SD3 随后达到接近的区间。该图用于说明预期训练
趋势，并不保证完全相同的最终分数；随机种子、prompt、初始 checkpoint、GPU 数量、
依赖版本和评测设置都会影响收敛速度与最终 reward。

健康的训练通常应表现为 ``env/avg`` 持续提高，同时 NFT update 相关统计保持在合理
范围。除了 scalar reward，也应检查实际生成样本：OCR 可以衡量文字匹配，但无法单独
反映视觉质量和视频时序一致性。

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

使用 TensorBoard 查看日志。如果在 ``env.*.video_cfg`` 中启用媒体保存，生成样本会
写入 ``video_base_dir``。

.. code:: bash

   tensorboard --host 0.0.0.0 --logdir logs/

Scalar 定义与 logger 目录结构见
:doc:`训练指标 </rst_source/reference/metrics>`。
