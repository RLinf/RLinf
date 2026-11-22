OpenPI 视觉短期记忆
===================

Short Memory 让 OpenPI π₀.₅ 在 SFT 和评测时读取近期的相机历史。历史信息先在视觉编码器中融入当前帧，再送入语言模型。

历史如何进入 Policy
-------------------

单张图像能显示碗放在盘子上，却未必能判断机器人此前是否已经举起过碗。Short Memory 把此前的画面也交给模型，供下一步动作使用。每个相机的输入从单帧变为同一 episode 内从旧到新的一组图像；state、prompt 和训练 action target 仍属于当前时刻。

这些帧先加入固定的相对正余弦时间编码（当前帧为零，历史帧按相对当前帧的秒数偏移），再一起经过 LayerNorm 和 SigLIP 原有的 Q/K/V 投影。在指定层中，每个 attention head 复用原有 SigLIP 参数，不引入新的可学习权重，按两步完成计算：

1. **时间混合**：固定每个 patch 位置，用因果 mask 沿帧维度聚合当前帧与过去的有效帧（padding 帧清零），得到带历史信息的 value。
2. **空间混合**：在每帧内部用原有 query 和 key 对上述历史 value 做全 patch attention，再经过原有输出投影、残差连接和 MLP。

默认在零起始的第 ``[3, 7, 11, 15]`` 层混合时间信息。第 15 层结束后，编码器丢弃历史帧，只保留已融合历史信息的当前帧特征，因此送入语言模型的视觉 token 数保持不变。SFT 仍用标准 flow-matching loss 更新网络权重。

选择历史窗口
------------

LIBERO-Mem 配置默认取 8 帧，包含当前帧，相邻帧间隔 70 步；在 20 Hz 下覆盖 24.5 秒。评测会记录每个控制步的图像，包括 action chunk 内部的步骤；SFT 在每条示范 episode 内采样。开头不足的历史补零并加上 mask，环境 reset 时只清空对应环境的历史。

SFT 的 ``num_frames`` 和 ``stride`` 放在 ``data.short_memory`` 下，评测则放在 ``env.eval.short_memory`` 下。评测用 ``fps`` 将步数差换算为秒，SFT 读取数据集时间戳；比较两者时应对齐实际时间跨度，尤其要检查数据转换是否删过帧。模型开关分别是 ``actor.model.openpi.short_memory`` 和 ``rollout.model.openpi.short_memory``，下方两个配置均已开启。时间混合层和丢弃历史的位置由 ``openpi.short_memory_temporal_layers`` 与 ``openpi.short_memory_drop_history_layer`` 设置。

在 LIBERO-Mem 上训练与评测
----------------------------------------

按 :doc:`OpenPI SFT <../sft_openpi>` 准备环境和初始权重，按 :doc:`LIBERO-Mem <../libero>` 准备模拟器。下面使用安装脚本默认的 ``.venv``。按 SFT 文档的下载方式准备 `LIBERO-Mem 示范数据 <https://huggingface.co/datasets/CollisionCode/libero_mem_lerobot_v2.1>`__ 和 `初始权重 <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__，再计算任务对应的归一化统计：

.. code-block:: bash

   python toolkits/lerobot/calculate_norm_stats.py --config-name pi05_libero_mem --repo-id CollisionCode/libero_mem_lerobot_v2.1

统计文件写入 ``assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json``。将初始 checkpoint 和该文件传给 SFT：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_mem_sft_openpi_pi05 \
     actor.model.model_path=./RLinf-Pi05-LIBERO-SFT \
     actor.model.openpi_data.norm_stats_path="$PWD/assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json"

该命令使用八帧历史做全参数 SFT。按 GPU 资源调整 ``examples/sft/config/libero_mem_sft_openpi_pi05.yaml`` 中的 batch 和更新步数。做当前帧对照时，保留 ``data.short_memory``，将 ``actor.model.openpi.short_memory=False``，即可使用同一套可恢复 loader。续训方式见 :doc:`Checkpoint Resume </rst_source/guides/resume>`。

用相同的归一化统计评测微调后的权重：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_mem_openpi_pi05_eval \
     rollout.model.model_path=./logs/checkpoints/global_step_1000/actor \
     rollout.model.openpi_data.norm_stats_path="$PWD/assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json"

默认评测 20 个环境、每条轨迹 700 个控制步。保留 ``env.eval.skip_intermediate_renders=False``，以记录 action chunk 内的历史。当前帧评测同时设置 ``rollout.model.openpi.short_memory=False`` 和 ``env.eval.short_memory=null``。结果与成功判定见 :doc:`LIBERO 评测 </rst_source/evaluations/guides/libero>`。

.. note::

   Short Memory 支持 ``openpi.task=eval`` 和 ``sft``。RL 训练的 replay 输入尚未携带相机历史，因此不支持该训练路径。
