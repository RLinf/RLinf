PhyAI 具身推理后端
==================

本页介绍如何用 `PhyAI <https://github.com/mingti-org/phyai>`_ 在进程内训练 PI0.5 policy，或通过 PhyAI Gateway 执行评估。PhyAI 是面向 VLA 等物理 AI 模型的推理框架。进程内路径由每个 RLinf rollout worker 管理一个 engine，LIBERO 环境、FSDP actor、观测变换和权重同步仍由 RLinf 负责。

下面先说明进程内 PPO 的安装与训练流程。需要通过 Gateway 评估时，请阅读 :ref:`phyai-http-evaluation-zh`。这条路径启动 RLinf 环境 worker 和 HTTP rollout worker，Gateway 和模型 Server 作为独立进程运行。

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

通过 PhyAI Gateway 评估
---------------------

这条路径仅用于评估，通过 Gateway 发送批量观测。不填写\ ``rollout.phyai.gateway_url``\ 时，RLinf 会在 placement 选中的 worker 上启动模型 Server 和 Gateway，并管理服务的生命周期。填写 Gateway 地址则使用外部管理的服务。

模型 Server 负责加载 checkpoint、预处理和动作后处理。RLinf 运行 LIBERO，并通过 HTTP 发送观测。

.. code-block:: text

   RLinf evaluation -> HTTP MessagePack -> PhyAI Gateway -> model server

使用评估环境
~~~~~~~~~~~~

使用\ :ref:`phyai-installation-zh`\ 一节创建的环境，其中已包含 PhyAI 模型 Server 和 Gateway。模型 Server 也可以运行在另一套环境或另一台机器上，两侧需使用兼容的 PhyAI、Gateway、checkpoint 和 tokenizer 版本。

下载 Checkpoint 和 Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gateway 路径使用 `LeRobot PI0.5 LIBERO checkpoint <https://huggingface.co/lerobot/pi05_libero_finetuned_v044>`_\ 和 PaliGemma tokenizer。保留 checkpoint 中的预处理、归一化和动作后处理文件。进程内训练使用的 RLinf/OpenPI checkpoint 布局不同，不能直接替代。

.. warning::

   `PaliGemma 仓库 <https://huggingface.co/google/paligemma-3b-pt-224>`_\ 受限。先在 Hugging Face 接受模型许可，再用\ ``hf auth login``\ 登录，或提供\ ``HF_TOKEN``\ 后下载。使用镜像也需要满足这些要求。

按下面的固定 revision 下载两份资源，将\ ``/path/to``\ 替换为实际存放目录；也可以使用已有的相同版本本地资源：

.. code-block:: bash

   hf auth login
   # 可选镜像：export HF_ENDPOINT=https://hf-mirror.com
   hf download lerobot/pi05_libero_finetuned_v044 \
     --revision 7152f13e34c6c115d808aea0e5d08aca74eac14b \
     --local-dir /path/to/pi05_libero_finetuned_v044
   hf download google/paligemma-3b-pt-224 \
     --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
     --local-dir /path/to/paligemma-3b-pt-224

配置 Gateway 服务
~~~~~~~~~~~~~~~~~

自动启动时，在\ ``evaluations/libero/libero_spatial_pi05_eval_phyai.yaml``\ 中填写以下路径。资源目录需能被 placement 选中的 worker 访问：

.. code-block:: yaml

   rollout:
     phyai:
       server_config: tests/e2e_tests/evaluations/phyai_gateway_server.yaml
       checkpoint_dir: /path/to/pi05_libero_finetuned_v044
       tokenizer_dir: /path/to/paligemma-3b-pt-224

RLinf 会写入临时 PhyAI Server 配置，启动两个服务，并等待 backend 健康。不填写\ ``gateway_url``\ 时，由 RLinf 管理服务，避免再手动启动重复进程。

使用外部 Gateway 时，将\ ``rollout.phyai.gateway_url``\ 设为其 HTTP 地址。资源路径由外部模型 Server 配置；此模式下 RLinf 不使用\ ``server_config``\、\ ``checkpoint_dir``\ 和\ ``tokenizer_dir``\。服务安装和启动步骤见\ `PhyAI Server 和 Gateway 文档 <https://phyai.mintlify.app/zh/deployment/server-gateway>`_\。

以下三处使用相同的模型名，例如\ ``pi05``\：

* 模型 Server 的\ ``server.model_name``\；
* Gateway\ ``--backend``\ 选项中的模型名；
* RLinf 的\ ``rollout.phyai.model_name``\。

运行评估
~~~~~~~~

设置好资源路径或外部 Gateway 地址后，在 RLinf 仓库根目录运行标准评估入口：

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai

该配置选择\ ``serving_mode: gateway``\，请求 16 步动作块。\ ``runner.only_eval``\ 必须保持为 true。

不要预先设置\ ``CUDA_VISIBLE_DEVICES``\，让 RLinf placement 选择目标 GPU。多 GPU 配置见\ :doc:`placement 指南 <../../evaluations/reference/configuration>`\。

在最终指标中检查\ ``success_once``\、\ ``success_at_end``\ 和\ ``num_trajectories``\。自动启动的服务由评估驱动停止；外部服务需在评估结束后手动停止。
