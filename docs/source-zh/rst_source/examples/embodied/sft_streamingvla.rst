StreamingVLA 监督微调
=========================

StreamingVLA 使用 Streaming Flow Policy（SFP），从 LIBERO 示教数据中学习动作
空间的速度场。本教程将依次准备公开的 LIBERO 数据集、计算归一化统计量、配置
Pi0.5 初始权重，并在 RLinf 中启动 FSDP2 全参数监督微调。


安装依赖
--------

在 RLinf 仓库根目录创建 StreamingVLA LIBERO 环境：

.. code:: bash

   bash requirements/install.sh embodied --model streamingvla --env libero
   source .venv/bin/activate


下载并转换 LIBERO 数据
----------------------

下载公开的 `OpenVLA modified LIBERO RLDS 数据集
<https://huggingface.co/datasets/openvla/modified_libero_rlds>`_：

.. code:: bash

   hf download openvla/modified_libero_rlds \
       --repo-type dataset \
       --local-dir /data/libero-rlds

设置 LeRobot 数据集的保存目录，并转换全部四个 LIBERO 任务套件：

.. code:: bash

   export HF_LEROBOT_HOME=/data/lerobot

   python toolkits/lerobot/convert_libero_data_to_lerobot.py \
       --data-dir /data/libero-rlds \
       --repo-name local/libero_streamingvla

转换后的数据位于 ``$HF_LEROBOT_HOME/local/libero_streamingvla``。转换脚本会在
每个 episode 中生成 SFP 所需的 ``action_states`` 字段。计算统计量和启动训练时，
应继续使用同一个 ``HF_LEROBOT_HOME``。


计算归一化统计量
----------------

创建资产根目录，然后根据转换后的数据计算统计量：

.. code:: bash

   mkdir -p /data/streamingvla/assets

   python toolkits/lerobot/calculate_streamingvla_norm_stats.py \
       --repo-id local/libero_streamingvla \
       --assets-dir /data/streamingvla/assets \
       --asset-id libero_streamingvla

本示例中的三个参数含义如下：

- ``repo_id`` 是 ``HF_LEROBOT_HOME`` 下的 LeRobot 数据集标识。
- ``assets_dir`` 是 OpenPI 格式数据资产的根目录。
- ``asset_id`` 是当前数据集在资产根目录中的子目录名称。

上述命令会生成
``/data/streamingvla/assets/libero_streamingvla/norm_stats.json``。该文件包含
``state``、``actions`` 和 ``action_states`` 的统计量。SFP 对 ``actions`` 和
``action_states`` 使用相同的归一化参数，因此工具会将完整的 ``actions`` 统计量
复制给 ``action_states``。


准备 Pi0.5 初始权重
-------------------

StreamingVLA 从官方 ``pi05_libero`` 权重开始微调。RLinf 需要一个包含
``model.safetensors`` 的 PyTorch checkpoint 目录。请在 `OpenPI
<https://github.com/Physical-Intelligence/openpi>`_ 仓库中下载并转换官方权重：

.. code:: bash

   python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"

   python examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero" \
       --config_name pi05_libero \
       --output_path /data/checkpoints/pi05_libero_pytorch

转换完成后，确认
``/data/checkpoints/pi05_libero_pytorch/model.safetensors`` 已生成。


配置并启动训练
--------------

修改 ``examples/sft/config/libero_sft_streamingvla.yaml``，填入前面生成的路径和
数据集标识：

.. code:: yaml

   data:
     train_data_paths: local/libero_streamingvla

   actor:
     model:
       model_path: /data/checkpoints/pi05_libero_pytorch
       streamingvla:
         data:
           repo_id: ${data.train_data_paths}
           assets:
             assets_dir: /data/streamingvla/assets
             asset_id: libero_streamingvla

配套配置使用 ``use_sfp: true``、``use_action_states: true``、动作 horizon 10、模型
动作维度 32、``sigma: 0.16`` 和 ``noise_decay: 4.0``，共训练 100,000 个
optimizer step，并每 5,000 步保存一次 checkpoint。

在 RLinf 仓库根目录启动训练：

.. code:: bash

   source .venv/bin/activate
   export HF_LEROBOT_HOME=/data/lerobot
   bash examples/sft/run_vla_sft.sh libero_sft_streamingvla

训练指标由 YAML 中配置的 logger 保存。Checkpoint 位于
``<runner.logger.log_path>/checkpoints/global_step_<N>/``。如需继续训练，将
``runner.resume_dir`` 指向一个完整的 ``global_step_<N>`` checkpoint 目录。
