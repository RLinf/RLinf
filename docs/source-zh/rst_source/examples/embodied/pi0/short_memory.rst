Short Memory
============

Short Memory 让 OpenPI π₀.₅ 在 SFT 和评测时读取近期的相机历史。历史信息先在视觉编码器中融入当前帧，再送入语言模型。

历史如何进入 Policy
-------------------

单张图像能显示碗放在盘子上，却未必能判断机器人此前是否已经举起过碗。Short Memory 把此前的画面也交给模型，供下一步动作使用。每个相机的输入从单帧变为同一 episode 内从旧到新的一组图像；state、prompt 和训练 action target 仍属于当前时刻。

这些帧一起进入 SigLIP。在指定层，先沿时间轴混合同一 patch 位置的历史特征，再在每帧内部混合不同 patch。下面按一个 attention head 说明这两步，省略 batch 维度。

设有 :math:`T` 帧、每帧 :math:`P` 个 patch，每个 head 的特征维度为 :math:`d`。帧索引 :math:`t=1,\ldots,T` 从旧到新，:math:`t=T` 是当前帧；:math:`p=1,\ldots,P` 表示 patch 位置。各帧先加入固定相对时间编码 :math:`e(\tau_t)=\mathrm{PE}(\tau_t)-\mathrm{PE}(0)`。PE 是固定的正余弦编码，:math:`\tau_t` 是相对当前帧的时间差，单位为秒；减去 PE(0) 使当前帧的时间编码为零。经过该层的 LayerNorm 和 SigLIP 原有投影，得到 :math:`Q,K,V\in\mathbb{R}^{T\times P\times d}`，分别表示 query、key 和 value。

Attention 用 Q 与 K 的相似度确定 V 的加权和：

.. math::

   \operatorname{Attn}(Q,K,V;M)
   = \operatorname{softmax}_{\mathrm{row}}\!\left(
       \frac{QK^\top}{\sqrt{d}}+M
     \right)V.

这里 softmax 对每个 query 对应的一行 key 分数归一化；:math:`1/\sqrt{d}` 缩放相似度，:math:`M` 是加性 mask：允许读取的位置为零，禁止的位置为 :math:`-\infty`，对应权重为零。

**时间混合。** 固定 patch 位置 :math:`p`，沿帧维度计算：

.. math::

   \widetilde V_{:,p,:}
   = \operatorname{Attn}(Q_{:,p,:},K_{:,p,:},V_{:,p,:};M_{\mathrm{time}}).

冒号表示保留该维度的全部元素，因此这里的输入都是 :math:`T\times d` 矩阵。对有效 query 帧 :math:`t`，:math:`M_{\mathrm{time}}[t,s]` 仅在 :math:`s\leq t` 且第 :math:`s` 帧有效时为零，其余为 :math:`-\infty`。也就是说，同一网格位置只读取当前和过去的有效帧，得到带历史信息的 :math:`\widetilde V`；padding 帧的输出在每层结束时清零。

**空间混合。** 固定帧 :math:`t`，沿 patch 维度计算：

.. math::

   O_{t,:,:}
   = \operatorname{Attn}(Q_{t,:,:},K_{t,:,:},\widetilde V_{t,:,:};0).

这一步的输入是 :math:`P\times d` 矩阵，帧内所有 patch 互相可见。Q、K 仍是最初的投影结果，只有 V 换成时间混合后的 :math:`\widetilde V`，而不是再从 :math:`\widetilde V` 生成一套 Q、K、V。各个 head 的输出 O 拼接后经过原有输出投影、残差连接和 MLP。整个过程复用 SigLIP 的参数，固定时间编码也没有可学习权重。

默认在零起始的第 ``[3, 7, 11, 15]`` 层混合时间信息。第 15 层结束后，只留下已融合历史信息的当前帧特征，交给剩余视觉层和语言模型。因此，送入语言模型的视觉 token 数保持不变，增加历史主要增加前半段的视觉计算量。SFT 仍用原有的 flow-matching loss 更新网络权重。

选择历史窗口
------------

LIBERO-Mem 配置默认取 8 帧，包含当前帧，相邻帧间隔 70 步；在 20 Hz 下覆盖 24.5 秒。评测会记录每个控制步的图像，包括 action chunk 内部的步骤；SFT 在每条示范 episode 内采样。开头不足的历史补零并加上 mask，环境 reset 时只清空对应环境的历史。

SFT 的 ``num_frames`` 和 ``stride`` 放在 ``data.short_memory`` 下，评测则放在 ``env.eval.short_memory`` 下。评测用 ``fps`` 将步数差换算为秒，SFT 读取数据集时间戳；比较两者时应对齐实际时间跨度，尤其要检查数据转换是否删过帧。模型开关分别是 ``actor.model.openpi.short_memory`` 和 ``rollout.model.openpi.short_memory``，下方两个配置均已开启。时间混合层和丢弃历史的位置由 ``openpi.short_memory_temporal_layers`` 与 ``openpi.short_memory_drop_history_layer`` 设置。

在 LIBERO-Mem 上训练与评测
----------------------------------------

按 :doc:`OpenPI SFT <../sft_openpi>` 准备环境和初始权重，按 :doc:`LIBERO-Mem <../libero>` 准备模拟器。下面使用安装脚本默认的 ``.venv``。按 SFT 文档的下载方式准备 `LIBERO-Mem 示范数据 <https://huggingface.co/datasets/CollisionCode/libero_mem_lerobot_v2.1>`__ 和 `初始权重 <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__，再计算任务对应的归一化统计：

.. code-block:: bash

   source .venv/bin/activate
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python toolkits/lerobot/calculate_norm_stats.py --config-name pi05_libero_mem --repo-id CollisionCode/libero_mem_lerobot_v2.1

统计文件写入 ``assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json``。将初始 checkpoint 和该文件传给 SFT：

.. code-block:: bash

   EMBODIED_PATH="$PWD/examples/sft" python examples/sft/train_vla_sft.py \
     --config-name libero_mem_sft_openpi_pi05 \
     actor.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT \
     actor.model.openpi_data.norm_stats_path="$PWD/assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json"

该命令使用八帧历史做全参数 SFT。按 GPU 资源调整 ``examples/sft/config/libero_mem_sft_openpi_pi05.yaml`` 中的 batch 和更新步数。做当前帧对照时，保留 ``data.short_memory``，将 ``actor.model.openpi.short_memory=False``，即可使用同一套可恢复 loader。续训方式见 :doc:`Checkpoint Resume </rst_source/guides/resume>`。

用相同的归一化统计评测微调后的权重：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_mem_openpi_pi05_eval \
     rollout.model.model_path=/path/to/short_memory_checkpoint \
     rollout.model.openpi_data.norm_stats_path="$PWD/assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json"

默认评测 20 个环境、每条轨迹 700 个控制步。保留 ``env.eval.skip_intermediate_renders=False``，以记录 action chunk 内的历史。当前帧评测同时设置 ``rollout.model.openpi.short_memory=False`` 和 ``env.eval.short_memory=null``。结果与成功判定见 :doc:`LIBERO 评测 </rst_source/evaluations/guides/libero>`。

.. note::

   Short Memory 支持 ``openpi.task=eval`` 和 ``sft``。RL 训练的 replay 输入尚未携带相机历史，因此不支持该训练路径。
