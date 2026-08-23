StreamingVLA 监督微调
=========================

本文档介绍 RLinf 中面向 LIBERO 示教数据的 **仅训练** StreamingVLA
集成。模型通过 ``model_type: streamingvla`` 注册，在 Pi0.5 主干上实现
Streaming Flow Policy（SFP）训练目标，并支持通过 FSDP2 进行全参数
SFT。本轮集成不包含动作生成、rollout、仿真评测或强化学习。

StreamingVLA 的代码隔离在
``rlinf/models/embodiment/streamingvla`` 中，不修改 RLinf 现有的
``openpi`` / ``openpi_rlinf`` 模型、全局 Transformers 包、通用归一化或
优化器实现。


安装依赖
--------

在仓库根目录中创建 StreamingVLA + LIBERO 环境：

.. code:: bash

   bash requirements/install.sh embodied --model streamingvla --env libero
   source .venv/bin/activate

与 ``--model openpi`` 的安装路径不同，StreamingVLA 不会将替换文件复制到全局
Transformers 目录；兼容的 Gemma、SigLIP、图像预处理和 SFP 逻辑都从
StreamingVLA 私有命名空间导入。


准备数据和权重
--------------

先将公开的 LIBERO RLDS 数据转换为 LeRobot 格式；转换器会直接写入每个时刻
动作执行前、从零起点累计得到的 ``action_states``：

.. code:: bash

   python toolkits/lerobot/convert_libero_data_to_lerobot.py \
       --data-dir /path/to/modified_libero_rlds \
       --repo-name your_hf_username/libero_streamingvla

如果同名输出数据集已经存在，转换器会安全失败；只有确实需要替换该数据集时才
显式传入 ``--overwrite``。

训练配方要求使用 map-style 的 LeRobot LIBERO 数据集。每个 episode 的真实
样本除了图像、腕部图像、机器人状态、动作和任务文本外，必须包含
``action_states`` 数组。dataloader 会直接读取第一条真实样本并检查该字段；
不会根据 metadata 推断它是否存在。

StreamingVLA 的局部 transform 将 LIBERO 的 7 维 state、action state 和动作
delta 填充到 32 维模型空间，并使用以零为中心的线性分位数归一化：

.. math::

   x / (\max(|q_{0.01}|, |q_{0.99}|) + 10^{-6}).

将 ``norm_stats.json`` 放在
``<assets_dir>/<asset_id>/norm_stats.json``。统计量必须包含数据集使用的
``state``、``actions`` 和 ``action_states`` 项。
可通过 StreamingVLA 专用工具生成：

.. code:: bash

   python toolkits/lerobot/calculate_streamingvla_norm_stats.py \
       --repo-id /path/to/libero_lerobot_dataset \
       --assets-dir /path/to/dataset-parent \
       --asset-id libero_lerobot_dataset

按照 SFP 训练约定，工具会把完整的 ``actions`` 统计数据直接复制给
``action_states``，因此 ``norm_stats.json`` 中这两项完全一致。

``actor.model.model_path`` 应指向含有 ``model.safetensors`` 的 Pi0.5
PyTorch 基础权重目录。OpenPI 旧格式基础权重和 RLinf StreamingVLA checkpoint
都由私有严格加载器处理；任何 missing key、unexpected key 或 shape mismatch
都会立即终止初始化。


配置训练
--------

提交的配置文件只包含可移植占位符：

- 实验配置：``examples/sft/config/libero_sft_streamingvla.yaml``
- 模型模板：``examples/sft/config/model/streamingvla.yaml``

至少在实验配置中替换以下路径：

.. code:: yaml

   data:
     train_data_paths: /path/to/libero_lerobot_dataset

   actor:
     model:
       model_path: /path/to/pi05_libero_pytorch
       streamingvla:
         data:
           repo_id: ${data.train_data_paths}
           assets:
             assets_dir: /path/to/dataset-parent
             asset_id: libero_lerobot_dataset

公开配置约定为 ``use_sfp: true``、``use_action_states: true``、horizon 10、
action dimension 32、``sigma: 0.16`` 和 ``noise_decay: 4.0``。默认配方使用
seed 42、global/micro batch 16/4、10,000 步 warmup、100,000 个 optimizer step，
并每 5,000 步保存 checkpoint。


启动训练
--------

在仓库根目录使用 RLinf 现有 VLA SFT 启动脚本：

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_streamingvla

默认 logger 是 TensorBoard。如需显式上传 Weights & Biases，先使用常规 W&B
环境完成登录，再只覆盖 logger backend 和 run 名称：

.. code:: bash

   export EMBODIED_PATH="$(pwd)/examples/sft"
   python examples/sft/train_vla_sft.py \
       --config-path examples/sft/config \
       --config-name libero_sft_streamingvla \
       'runner.logger.logger_backends=[wandb]' \
       runner.logger.experiment_name=streamingvla_libero_sft

RLinf 会将 checkpoint 写入
``<runner.logger.log_path>/checkpoints/global_step_<N>/``。如需续训，应通过
``runner.resume_dir`` 指向一个完整的 RLinf checkpoint 目录。如果调用推理、
rollout 或非 SFT forward，本集成会明确抛出 ``NotImplementedError``。
