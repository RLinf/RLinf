ABot-M0 强化学习训练
====================

本文档介绍如何在 RLinf 中对 `ABot-M0 <https://github.com/amap-cvlab/ABot-Manipulation>`__ 进行评测与 PPO 训练。示例配置覆盖标准 **LIBERO** 和 **LIBERO-Plus**。

该适配使用 Hugging Face rollout backend 和 FSDP actor 训练。rollout 阶段，ABot-M0 为 LIBERO 环境生成动作块；actor 更新阶段，RLinf 基于 rollout 中保存的输入重新计算 log probability 和 value。

算法
----

本示例使用 actor-critic 形式的 PPO：

* 使用 GAE 估计 advantage 和 return。
* 使用 PPO ratio clipping 约束策略更新。
* 对 value head 使用 value-function clipping。
* 支持可选的 entropy regularization。

ABot-M0 作为 VLA 策略接入 RLinf。适配层冻结预训练感知模块，通过 RL objective 训练动作模型，并额外加入 value head 以支持 actor-critic 训练。

依赖安装
--------

请在同一个 Python 环境中安装 ABot-M0、VGGT 和 LIBERO 运行时。

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

请通过镜像内置的 ``switch_env`` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env abot_m0

**选项 2：自定义环境**

安装脚本会自动克隆 ABot-M0 和 VGGT。如果你已经有本地源码 checkout，可在运行安装脚本前设置
``ABOT_PATH`` 和 ``VGGT_PATH``。

.. code:: bash

   # 可选：使用本地源码 checkout，而不是由安装脚本自动克隆。
   # export ABOT_PATH=<path_to_ABot-Manipulation>
   # export VGGT_PATH=<path_to_vggt>

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 到下面的 install.sh 命令。
   bash requirements/install.sh embodied --model abot_m0 --env maniskill_libero
   source .venv/bin/activate

如果需要运行 LIBERO-Plus 实验，请在同一环境中额外安装 ``LIBERO-plus`` 运行时：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 到下面的 install.sh 命令。
   bash requirements/install.sh embodied --model abot_m0 --env liberoplus
   source .venv/bin/activate

LIBERO-Plus 资产下载
--------------------

LIBERO-Plus 需要大量新增对象、纹理和其他资产才能正常运行。请从 Hugging Face dataset
``Sylvest/LIBERO-plus`` 下载 ``assets.zip``，并解压到已安装的
``liberoplus.liberoplus`` package 目录：

.. code-block:: bash

   # 获取已安装的 liberoplus 包目录。
   # 注意：导入 liberoplus 时可能会触发配置初始化日志，因此使用 tail -n 1 只保留最终路径。
   export LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)" | tail -n 1)

   echo "LIBERO_PLUS_PACKAGE_DIR=${LIBERO_PLUS_PACKAGE_DIR}"

   # 如果运行环境无法直接访问 Hugging Face，可启用镜像。
   # export HF_ENDPOINT=https://hf-mirror.com

   # 从 Hugging Face dataset 仓库下载资产压缩包。
   hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip \
       --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"

   # assets.zip 内部包含较长的原始路径前缀，因此只提取其中 assets/ 下的内容。
   python - <<'PY'
   import zipfile
   from pathlib import Path

   pkg = Path(__import__("os").environ["LIBERO_PLUS_PACKAGE_DIR"])
   zip_path = pkg / "assets.zip"
   out_dir = pkg / "assets"

   with zipfile.ZipFile(zip_path) as z:
       for info in z.infolist():
           name = info.filename

           if "/assets/" not in name:
               continue

           rel = name.split("/assets/", 1)[1]
           if not rel:
               continue

           target = out_dir / rel

           if info.is_dir():
               target.mkdir(parents=True, exist_ok=True)
           else:
               target.parent.mkdir(parents=True, exist_ok=True)
               with z.open(info) as src, open(target, "wb") as dst:
                   dst.write(src.read())

   print("Extracted LIBERO-Plus assets to:", out_dir)
   PY

   # 检查资产目录结构。
   ls -lh "${LIBERO_PLUS_PACKAGE_DIR}/assets"

解压完成后，目录应类似如下：

.. code-block:: text

   <已安装的 liberoplus 包目录>/
   └── assets/
       ├── articulated_objects/
       ├── new_objects/
       ├── scenes/
       ├── stable_hope_objects/
       ├── stable_scanned_objects/
       ├── textures/
       ├── turbosquid_objects/
       ├── serving_region.xml
       ├── wall_frames.stl
       └── wall.xml

LIBERO-Plus 的完整说明见 LIBERO 基准文档的 :ref:`LIBERO-Pro 与 LIBERO-Plus 章节 <zh-liberopro-plus-benchmark>`。

模型下载
--------

训练开始前，请下载 ABot-M0 checkpoint 和所需 backbone 权重：

* ``acvlab/ABot-M0-LIBERO``：用于独立评测的 SFT 权重。
* ``HaoyunOvO/ABot-m0-LIBERO-10k-step``：用于 PPO 训练的 RL baseline。
* ``StarVLA/Qwen3-VL-4B-Instruct-Action``：Qwen3-VL backbone。
* ``facebook/VGGT-1B``：运行时无法访问 Hugging Face 时用于离线加载 VGGT。

.. code-block:: bash

   # 方式 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/acvlab/ABot-M0-LIBERO
   git clone https://huggingface.co/HaoyunOvO/ABot-m0-LIBERO-10k-step
   git clone https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action
   git clone https://huggingface.co/facebook/VGGT-1B

   # 方式 2：使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download acvlab/ABot-M0-LIBERO --local-dir ./ABot-M0-LIBERO
   hf download HaoyunOvO/ABot-m0-LIBERO-10k-step --local-dir ./ABot-m0-LIBERO-10k-step
   hf download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir ./Qwen3-VL-4B-Instruct-Action
   hf download facebook/VGGT-1B --local-dir ./VGGT-1B

PPO 训练可使用 10k-step ABot-M0 LIBERO checkpoint 作为 RL baseline。该权重在 LIBERO
评测中的初始成功率约为 40%，适合作为后续 RL 训练的起点。

.. note::

   ABot-M0 checkpoint 自带 ``config.yaml``。下载完成后，请修改 ``qwenvl.base_vlm``，
   使其指向本机 ``Qwen3-VL-4B-Instruct-Action`` 路径。

.. code-block:: yaml

   qwenvl:
     base_vlm: /path/to/Qwen3-VL-4B-Instruct-Action

ABot 当前默认使用 ``VGGT.from_pretrained("facebook/VGGT-1B")`` 初始化 VGGT。如果运行时无法访问
Hugging Face 或镜像，请将 ``VGGT-1B`` 放入本地 Hugging Face cache，或在 ABot 安装代码中将
VGGT 加载路径显式改为本地目录。

本地路径示例：

.. code-block:: python

   self.spatial_model = spatial_model = VGGT.from_pretrained('/workspace/models/VGGT-1B')

配置 ``model_path``
-------------------

针对两个 benchmark 各提供一份配置：

* LIBERO：      ``examples/embodiment/config/libero_10_ppo_abot_m0.yaml``
* LIBERO-Plus： ``examples/embodiment/config/libero_10_plus_ppo_abot_m0.yaml``

请将以下两项设置为用于评测或训练的 checkpoint 路径：

* ``rollout.model.model_path``
* ``actor.model.model_path``

如果使用 10k-step RL baseline，请设置为：

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/ABot-m0-LIBERO-10k-step/checkpoints/steps_10000_pytorch_model.pt
   actor:
     model:
       model_path: /path/to/ABot-m0-LIBERO-10k-step/checkpoints/steps_10000_pytorch_model.pt

导入完整性验证
--------------

.. code-block:: bash

   python -c "import rlinf; import ABot; import vggt; print('IMPORT_OK')"

若输出 ``IMPORT_OK``，说明包级依赖链路正常。

评测
----

建议在训练前先执行独立评测，用于验证 checkpoint、rollout 流程和环境资产是否正确。

评测入口是 ``evaluations/eval_embodied_agent.py``。两个 benchmark
共用同一套启动流程，差异只在 ``LIBERO_TYPE`` 与配置文件名。

通用环境变量：

.. code-block:: bash

   source .venv/bin/activate

   export REPO_PATH=$(pwd)
   export EMBODIED_PATH=$(pwd)/examples/embodiment
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export ROBOT_PLATFORM=LIBERO

   ray stop || true
   ray start --head --port=6379

**LIBERO：**

.. code-block:: bash

   export LIBERO_TYPE=standard

   python evaluations/eval_embodied_agent.py \
     --config-name libero_10_ppo_abot_m0 \
     actor.model.model_path=<path_to_abot_m0_ckpt> \
     rollout.model.model_path=<path_to_abot_m0_ckpt> \
     runner.only_eval=True \
     env.eval.total_num_envs=8 \
     env.eval.video_cfg.save_video=true \
     env.eval.rollout_epoch=1 \
     runner.logger.experiment_name=abot_m0_libero10_eval

**LIBERO-Plus：**

.. code-block:: bash

   export LIBERO_TYPE=plus

   python evaluations/eval_embodied_agent.py \
     --config-name libero_10_plus_ppo_abot_m0 \
     actor.model.model_path=<path_to_abot_m0_ckpt> \
     rollout.model.model_path=<path_to_abot_m0_ckpt> \
     runner.only_eval=True \
     env.eval.total_num_envs=8 \
     env.eval.video_cfg.save_video=true \
     env.eval.rollout_epoch=1 \
     runner.logger.experiment_name=abot_m0_liberoplus_eval

训练
----

PPO 训练与评测共用同一套启动流程。通过 ``LIBERO_TYPE`` 选择目标套件，并启动对应配置。

通用环境变量：

.. code-block:: bash

   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   export EMBODIED_PATH=$(pwd)/examples/embodiment
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export ROBOT_PLATFORM=LIBERO

   ray stop || true
   ray start --head --port=6379

**LIBERO：**

.. code-block:: bash

   export LIBERO_TYPE=standard
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_abot_m0

**LIBERO-Plus：**

.. code-block:: bash

   export LIBERO_TYPE=plus
   bash examples/embodiment/run_embodiment.sh libero_10_plus_ppo_abot_m0

可视化
------

.. code-block:: bash

   tensorboard --logdir <runner.logger.log_path> --port 6006
