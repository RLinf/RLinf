OPD：OpenVLA-OFT 在线策略蒸馏
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

OPD（On-Policy Distillation）在 student 策略自己的 on-policy rollout 上做蒸馏：student
先与环境交互并保存动作 token，rollout 结束后 teacher 只对这些 student action tokens 计算
逐 token 的 log probability。RLinf 中的 OPD 不在 rollout 阶段生成 teacher action；teacher 只在
rollout 后打分，完成后立即 offload，以减少 rollout 延迟和显存占用。

概览
----------------------------------------

本示例在 **LIBERO-Spatial** 上使用 **OpenVLA-OFT student** 与 **OpenVLA-OFT teacher** 进行 OPD 训练。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO-Spatial

   .. grid-item-card:: 算法
      :text-align: center

      OPD

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT student · OpenVLA-OFT teacher

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 张 GPU

| **你将完成：** 下载 student/teacher 权重 -> 检查 ``unnorm_key`` -> 启动 ``libero_spatial_opd_openvlaoft`` -> 观察 ``env/success_once`` 与 OPD actor 指标。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · :doc:`LIBERO 环境 <libero>` · 已可运行 OpenVLA-OFT + LIBERO。

方法要点
----------------------------------------

OPD 的训练信号来自 teacher 与 student 对同一条 student action token 序列的相对 logprob：

.. math::

   A_{OPD} = \log \pi_{teacher}(a_{student} \mid s) - \operatorname{stop\_grad}(\log \pi_{student}(a_{student} \mid s))

实现上需要注意三点：

- student rollout 是唯一的环境交互来源；teacher 不参与采样动作。
- teacher 在 rollout 之后对保存的 ``forward_inputs.action_tokens`` 计算 dense token logprob。
- OPD loss 使用 ``loss_mask`` 屏蔽成功或终止后的无效 token，避免无效 token 继续贡献梯度。

权重
----------------------------------------

.. list-table:: **LIBERO-Spatial OPD 使用的权重**
   :header-rows: 1
   :widths: 18 42 26 28

   * - 角色
     - Hugging Face 权重
     - 配置字段
     - ``unnorm_key``
   * - Student
     - |huggingface| `RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`_
     - ``actor.model.model_path`` / ``rollout.model.model_path``
     - ``libero_130_no_noops_trajall``
   * - Teacher
     - |huggingface| `RLinf/RLinf-OpenVLAOFT-LIBERO-130 <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130>`_
     - ``rollout.expert_model.model_path``
     - ``libero_130_no_noops_trajall``

下载权重（任选一种方式）：

.. code-block:: bash

   # 方法 1：git-lfs
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130

   # 方法 2：huggingface-hub（国内可设置 HF_ENDPOINT=https://hf-mirror.com）
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130 --local-dir RLinf-OpenVLAOFT-LIBERO-130

配置
----------------------------------------

示例配置位于 ``examples/embodiment/config/libero_spatial_opd_openvlaoft.yaml``。关键字段如下：

.. code-block:: yaml

   algorithm:
     adv_type: opd
     loss_type: opd
     loss_agg_func: token-mean

   actor:
     global_batch_size: 4096
     micro_batch_size: 32
     model:
       model_path: /path/to/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
       unnorm_key: libero_130_no_noops_trajall
     optim:
       lr: 2.0e-6

   rollout:
     model:
       model_path: /path/to/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
     expert_model:
       model_path: /path/to/RLinf-OpenVLAOFT-LIBERO-130
       unnorm_key: libero_130_no_noops_trajall

   env:
     train:
       rollout_epoch: 8
       max_episode_steps: 240
       max_steps_per_rollout_epoch: 240

.. note::

   Student 与 teacher 必须使用一致的 ``unnorm_key``。本示例中的两个权重都使用
   ``libero_130_no_noops_trajall``。

运行
----------------------------------------

使用脚本启动完整训练：

.. code-block:: bash

   source switch_env openvla-oft
   bash examples/embodiment/run_embodiment.sh libero_spatial_opd_openvlaoft


训练日志写入 ``runner.logger.log_path``，默认使用 TensorBoard：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

重点关注指标：

- ``env/success_once``：未归一化的回合成功率。
- ``actor/opd_reverse_kl``：teacher 与 student 在 student action tokens 上的 logprob 差异。
- ``actor/opd_reward``：OPD 使用的 token-level 蒸馏 reward。
- ``actor/policy_loss`` 与 ``actor/total_loss``：actor 更新是否数值稳定。

结果
----------------------------------------

下表先给出 LIBERO-Spatial 上的结果。Base student 为
|huggingface| `RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`_；
teacher 为 |huggingface| `RLinf/RLinf-OpenVLAOFT-LIBERO-130 <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130>`_。

.. list-table:: **LIBERO-Spatial success rate**
   :header-rows: 1
   :widths: 40 24 36

   * - 模型 / 训练阶段
     - Success rate
     - 备注
   * - OpenVLA-OFT LIBERO-130 Base-Lora student
     - 约 67%
     - OPD 训练前的 student baseline。
   * - OPD training, 30 steps
     - 96%+
     - 使用 ``libero_spatial_opd_openvlaoft`` 配置得到的早期训练结果。
