ApxInf Rollout 后端
======================

使用 `ApxInf <https://github.com/infinigence/ApxInf>`_ 作为进程内 Rust/CUDA
rollout 后端，在 LIBERO 上评测 π₀.₅ 策略。RLinf 通过提供稳定 L1 bare-model
接口的 `APXinf-robo <https://github.com/RLinf/APXinf-robo>`_ 接入引擎。RLinf
负责环境交互、batch、OpenPI 预处理与后处理以及动作执行，ApxInf 负责 GPU
模型推理。

.. note::

   当前接入仅支持在 LIBERO 上评测 π₀.₅，不支持训练、解耦 rollout 或非具身任务。

前置条件
--------

在执行评测的机器上准备以下内容：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 要求
     - 说明
   * - 运行平台
     - 配有 NVIDIA GPU、CUDA toolkit 和稳定版 Rust toolchain 的 Linux 系统。
   * - RLinf 环境
     - 安装 OpenPI 模型与 LIBERO 环境依赖。
   * - APXinf-robo 与 ApxInf
     - 安装 APXinf-robo，并在目标机器上为其 ApxInf 子模块构建对应 GPU 架构的
       CUDA Python binding。将两者安装到 RLinf 所在的同一环境。
   * - Checkpoint
     - π₀.₅ LIBERO checkpoint，其中包含 ``config.json``、
       ``model.safetensors``、``tokenizer.model`` 和 ``norm_stats.json``。

安装 RLinf 与 APXinf-robo
-------------------------

安装 RLinf 的 OpenPI 与 LIBERO 依赖，然后激活生成的环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero
   source .venv/bin/activate

递归克隆 APXinf-robo，并将 ApxInf CUDA binding、ApxInf Python 前端和
APXinf-robo wrapper 安装到当前环境：

.. code-block:: bash

   git clone --recursive https://github.com/RLinf/APXinf-robo.git
   cd APXinf-robo
   pip install maturin
   CARGO_TARGET_DIR=target/wheel maturin build --release --features cuda \
     --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml
   pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl
   pip install -e ./apxinf/python/apxinf --config-settings editable_mode=strict
   pip install -e ".[rlinf]"

CUDA 构建默认面向构建机器上检测到的 GPU。如需部署到其他架构，请在运行
``maturin`` 前设置 ``APXINF_CUDA_ARCH``。支持的架构与 wheel 打包方式见
`APXinf-robo 构建指南
<https://github.com/RLinf/APXinf-robo#build-apxinf-robo>`_。

在 RLinf 环境中验证 binding：

.. code-block:: bash

   python -c 'import apxinf_py; print(apxinf_py.__version__)'
   python -c 'import apxinf_robo; print(apxinf_robo.__version__)'

准备 Checkpoint
---------------

下载 π₀.₅ LIBERO checkpoint，并通过环境变量将目录传给示例配置：

.. code-block:: bash

   pip install -U huggingface_hub
   hf download lerobot/pi05_libero_base \
     --local-dir /path/to/pi05_libero_base
   curl -fL \
     https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
     -o /path/to/pi05_libero_base/norm_stats.json
   export APXINF_PI05_MODEL_DIR=/path/to/pi05_libero_base

需要按上面的命令单独下载 ``norm_stats.json``。当前
``lerobot/pi05_libero_base`` 仓库不包含复现参考评测行为所需的归一化统计。

运行评测
--------

回到 RLinf 仓库根目录，运行已有的 LIBERO-10 配置：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_10_apxinf_pi05_eval

默认配置使用 10 个并行环境和 10 个 rollout epoch，对 LIBERO-10 的每个任务
评测 10 个 reset state，共 100 条轨迹。终端会输出 ``eval/success_once``；评测
协议和结果位置见 :doc:`LIBERO 评测 <../evaluations/guides/libero>`。

配置后端
--------

以 ``evaluations/libero/libero_10_apxinf_pi05_eval.yaml`` 为起点。关键配置如下：

.. code-block:: yaml

   rollout:
     rollout_backend: apxinf
     generation_backend: apxinf
     model:
       model_path: ${oc.env:APXINF_PI05_MODEL_DIR}
       num_action_chunks: 5
       action_dim: 7
       openpi:
         config_name: pi05_libero
         num_steps: 5
       apxinf:
         precision: bf16
         action_horizon: 10
         num_flow_steps: ${..openpi.num_steps}
         noise_source: apxinf

保持 ``runner.only_eval: true`` 和
``runner.enable_decoupled_mode: false``。``num_flow_steps`` 必须与
``openpi.num_steps`` 一致。默认的 ``noise_source: apxinf`` 由 ApxInf 采样初始
高斯 noise。仅在成对数值对齐测试中使用 ``noise_source: observation``，并显式
传入 ``[B, 10, 32]`` noise tensor。

接入原理
--------

RLinf 负责 LIBERO 交互、Ray 调度、观测 batch、图像 resize、tokenize、checkpoint
归一化、action 反归一化与动作块执行。Adapter 调用
``apxinf_robo.load_bare_model``，由它选择调优后的 GEMM tactics 并打开 ApxInf
底层 ``Model.infer_rgb`` API，再将处理后的 RGB 图像、token ID 和可选 noise
传给引擎。ApxInf 返回模型空间中的归一化 action，再由 RLinf 转换为环境 action。
