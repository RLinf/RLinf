PhyAI 具身推理后端
==================

`PhyAI <https://github.com/mingti-org/phyai>`_ 是面向 VLA 等具身模型的推理框架。在 RLinf 中，PhyAI 支持两种用法：

- **PPO 训练加速**：在每个 rollout worker 内直接运行 PhyAI 推理引擎负责动作采样，LIBERO 环境、FSDP actor、观测变换和权重同步仍由 RLinf 管理。
- **独立 HTTP 服务评测**：将 PI0.5 模型部署为独立的推理服务，RLinf 只运行 LIBERO 环境并通过 HTTP 请求获取动作（见 :ref:`phyai-http-evaluation-zh`）。

准备工作
--------

需要一台配备 NVIDIA GPU 的 Linux 机器和 Python 3.12。该后端支持在 LIBERO 上用同步 PPO 训练 PI0.5，每个 rollout worker 使用一张 GPU。训练流程已使用 RLinf PI0.5 LIBERO SFT checkpoint，在两张 NVIDIA H800 80 GB GPU 上完成快速验证。

.. _phyai-installation-zh:

安装
----

在 RLinf 仓库根目录下，将 RLinf、OpenPI、LIBERO 和 PhyAI 安装到同一个虚拟环境，然后激活该环境：

.. code-block:: bash

   bash requirements/install.sh \
     --python 3.12 \
     --install-rlinf \
     embodied \
     --model openpi \
     --env libero \
     --engine phyai
   source .venv/bin/activate

这条命令做了两件事：

1. 按 Hugging Face rollout 的方式安装 RLinf、OpenPI 和 LIBERO。
2. 按已安装 Torch 的 CUDA 版本，安装 RLinf 测试过的 PhyAI 版本。PhyAI 自带的依赖会替换 OpenPI 所需的 Torch 和 Transformers 版本，因此安装器保留原有版本；如果 PhyAI 的 CUDA 版本与 Torch 不一致，安装会直接停止。

准备 Checkpoint
---------------

先下载 PI0.5 LIBERO SFT checkpoint：

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir RLinf-Pi05-LIBERO-SFT

然后在 ``examples/embodiment/config/libero_spatial_ppo_openpi_pi05_phyai.yaml`` 中把 ``actor.model.model_path`` 设为下载目录。rollout 读取同一路径，因此 PhyAI 与 actor 加载相同的权重，并用同一份统计量归一化观测。其他 PI0.5 checkpoint 及下载镜像见 :doc:`π₀ 与 π₀.₅ 模型强化学习训练 <../../examples/embodied/pi0>`。

如果要从 RL 训练得到的 checkpoint 继续，还需将 ``runner.ckpt_path`` 设为对应的 ``.pt`` 文件。每个 PhyAI engine 会在下载的 checkpoint 之上加载这份 state dict，因此第一次 rollout 使用的权重与 actor 一致，冻结参数也不例外。

配置后端
--------

示例配置已经选择了 PhyAI，关键的几行如下：

.. code-block:: yaml

   defaults:
     - hybrid_engines/phyai@rollout.phyai

   algorithm:
     recompute_logprobs: True
     importance_sampling_fix: True
     importance_sampling_clip: 1.0

   rollout:
     rollout_backend: phyai
     phyai:
       max_batch_size: null

``rollout.rollout_backend: phyai`` 让训练脚本启动 PhyAI rollout worker，而不是 Hugging Face rollout worker。``hybrid_engines/phyai`` 提供 kernel 策略，固定 PhyAI 在 PI0.5 上使用的 attention 和激活函数 kernel，使其数值在两种实现允许的范围内尽量接近 actor。``max_batch_size`` 限制单个 engine 一次处理的观测数量；设为 ``null`` 时，按分配给该 worker 的环境数量确定。

``algorithm`` 中的三个配置项用于处理 PhyAI 与 actor 之间剩余的数值差异，下一节说明原因。

修正 Rollout 与 Actor 的差异
----------------------------

PhyAI 和 FSDP actor 使用同一份权重，但计算所用的 kernel 不同，例如 paged attention、融合的激活函数和不同的归约顺序。因此，actor 为 PhyAI 采样出的动作计算 log-probability 时，结果会与 PhyAI 报告的值略有出入。

PPO 以产生数据的 policy 为基准衡量每次更新。如果这个基准来自 PhyAI，kernel 带来的差异就会进入每一个 PPO ratio，甚至单独触发 clipping。开启 ``recompute_logprobs`` 后，actor 会在训练前为每个采样动作重新计算 log-probability，并以自己的结果作为基准。``importance_sampling_fix`` 再按 actor 与 PhyAI 对同一动作的概率之比为每个样本加权，权重上限为 ``importance_sampling_clip``，从而让 actor 能正确使用并非由自己采样的数据。这三个配置项需要一起开启：只开启 ``importance_sampling_fix`` 而未开启 ``recompute_logprobs`` 时，启动阶段会报错。

修正方法及其指标的详细说明见 :doc:`PPO <../../reference/algorithms/ppo>`。

启动训练
--------

设置好 checkpoint 路径后，在仓库根目录启动示例：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
     libero_spatial_ppo_openpi_pi05_phyai

该命令通过具身训练入口启动 FSDP actor、PhyAI rollout worker 和 LIBERO 环境 worker。每个训练 step 先由 PhyAI 收集 rollout，再由 actor 重新计算 log-probability 并更新参数，最后在下一轮 rollout 前把新权重同步到所有 PhyAI engine。

检查训练结果
------------

启动脚本将日志写入 ``logs/<timestamp>-<config-name>/``。``env/success_once`` 反映任务成功率。``actor/importance_sampling_weight`` 接近 1、``actor/importance_sampling_clip_fraction`` 较小，说明 PhyAI 与 actor 的结果仍然一致；如果权重逐渐偏离 1，说明二者的差异在扩大。其余指标的定义见 :doc:`训练指标 <../../reference/metrics>`。

.. _phyai-http-evaluation-zh:

通过 PhyAI HTTP 服务运行评测
----------------------------

评测模式下，PI0.5 模型运行在独立的推理服务进程中，与 RLinf 的 LIBERO 环境解耦。RLinf 负责运行环境并将批量观测通过 HTTP 发给 PhyAI Gateway，由后端模型服务完成权重加载、观测预处理和动作后处理。

.. code-block:: text

   RLinf evaluation -> HTTP MessagePack -> PhyAI Gateway -> model server

默认情况下，RLinf 会在本机自动拉起模型服务和 Gateway，评测结束后自动关闭；如果要对接已经部署好的外部服务，直接指定 ``rollout.phyai.gateway_url`` 即可。

环境与硬件要求
~~~~~~~~~~~~~~

需要至少一张 NVIDIA GPU，直接使用 :ref:`phyai-installation-zh` 中安装好的虚拟环境即可（已包含模型服务和 Gateway）。

默认配置会使用当前机器上的全部 GPU，并在第一张 GPU 上启动一组模型服务和 Gateway。如果想为推理服务单独指定一张或多张 GPU，可在 ``cluster.component_placement`` 中配置 ``phyai_server``：每张指定 GPU 启动一组服务，由各 rollout worker 轮询调用。语法见 :doc:`Placement <../../concepts/placement>`。

下载 Checkpoint 和 Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HTTP 服务评测使用 `LeRobot PI0.5 LIBERO checkpoint <https://huggingface.co/lerobot/pi05_libero_finetuned_v044>`_ 和 PaliGemma tokenizer，格式与前面 PPO 训练使用的 RLinf/OpenPI checkpoint 不同，不能混用。

.. warning::

   `PaliGemma 仓库 <https://huggingface.co/google/paligemma-3b-pt-224>`_ 需要申请访问权限。下载前先在 Hugging Face 页面接受许可协议，再运行 ``hf auth login`` 或设置 ``HF_TOKEN``。使用镜像源同样需要完成授权。

按以下已验证的 revision 下载 checkpoint 和 tokenizer：

.. code-block:: bash

   hf download lerobot/pi05_libero_finetuned_v044 \
     --revision 7152f13e34c6c115d808aea0e5d08aca74eac14b \
     --local-dir ./pi05_libero_finetuned_v044
   hf download google/paligemma-3b-pt-224 \
     --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
     --local-dir ./paligemma-3b-pt-224

自动启动并运行评测
~~~~~~~~~~~~~~~~~~

激活 PhyAI 虚拟环境后，在 RLinf 仓库根目录传入 checkpoint 和 tokenizer 路径即可启动 LIBERO-Spatial 评测：

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai \
     rollout.phyai.checkpoint_dir=./pi05_libero_finetuned_v044 \
     rollout.phyai.tokenizer_dir=./paligemma-3b-pt-224

启动后，脚本会在本机拉起模型服务和 Gateway，用 128 个并行 LIBERO-Spatial 环境评测 1 个 epoch（16 步动作块），结束后自动关闭服务。单次推理请求默认按 ``rollout.phyai.max_batch_size: 8`` 切分，1、2、4、8 张 GPU 均可直接运行。

如果不想每次在命令行传入路径，可以直接修改 ``evaluations/libero/libero_spatial_pi05_eval_phyai.yaml`` 中的 ``checkpoint_dir`` 和 ``tokenizer_dir``。

连接外部服务
~~~~~~~~~~~~

如果模型服务和 Gateway 已经在其他环境或机器上启动，直接通过 ``rollout.phyai.gateway_url`` 指定地址即可：

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai \
     rollout.phyai.gateway_url=http://127.0.0.1:30000

此时模型路径和 tokenizer 由外部模型服务自行配置。注意保持模型服务的 ``server.model_name``、Gateway 的 ``--backend`` 参数以及 RLinf 的 ``rollout.phyai.model_name`` 一致（默认均为 ``pi05``）。独立部署方式见 `PhyAI 服务与 Gateway 文档 <https://phyai.mintlify.app/zh/deployment/server-gateway>`_。

查看评测结果
~~~~~~~~~~~~

评测结束后，终端会输出最终指标表（包含 ``eval/success_once``、``eval/success_at_end`` 和 ``eval/num_trajectories``），日志与 TensorBoard 文件保存在 ``logs/<timestamp>-libero_spatial_pi05_eval_phyai/``。自动启动的模型服务和 Gateway 日志位于该目录的 ``phyai/`` 子目录下。LIBERO 评测协议见 :doc:`LIBERO 评测 <../../evaluations/guides/libero>`。
