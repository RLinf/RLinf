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

安装
--------

依赖安装
~~~~~~~~

请在同一个 Python 环境中安装 ABot-M0、VGGT 和标准 LIBERO 运行时。下面的 installer 命令会安装标准 LIBERO；只有 LIBERO-Plus 需要执行下一节中的额外步骤。

.. code-block:: bash

   git clone https://github.com/amap-cvlab/ABot-Manipulation.git
   git clone https://github.com/facebookresearch/vggt.git

   cd <path_to_RLinf>
   export ABOT_PATH=<path_to_ABot-Manipulation>
   export VGGT_PATH=<path_to_vggt>

   # 如果运行环境访问 GitHub 或 PyPI 较慢，可在安装命令中加入 --use-mirror。
   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env maniskill_libero --install-rlinf
   source .venv/bin/activate

LIBERO-Plus
~~~~~~~~~~~

LIBERO-Plus 需要额外安装 ``LIBERO-plus`` 仓库。在上述标准 LIBERO 安装完成后，使用
``--env liberoplus`` 在同一环境中追加安装：

.. code-block:: bash

   cd <path_to_RLinf>

   # 如果运行环境访问 GitHub 或 PyPI 较慢，可在安装命令中加入 --use-mirror。
   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env liberoplus --install-rlinf
   source .venv/bin/activate

随后将 LIBERO-Plus 资产下载并解压到已安装的 package 目录：

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

LIBERO-Plus 的完整说明见 :doc:`liberoplus_pro`。

模型权重
--------

评测权重
~~~~~~~~

ABot 发布的 SFT 权重可用于独立评测：

.. code-block:: bash

   # 如果运行环境无法直接访问 Hugging Face，可启用镜像。
   # export HF_ENDPOINT=https://hf-mirror.com

   hf download acvlab/ABot-M0-LIBERO --local-dir /path/to/ABot-M0-LIBERO

RL baseline 权重
~~~~~~~~~~~~~~~~

PPO 训练可使用 10k-step ABot-M0 LIBERO checkpoint 作为 RL baseline。该权重在 LIBERO 评测中的初始成功率约为 40%，适合作为后续 RL 训练的起点。

.. code-block:: bash

   hf download HaoyunOvO/ABot-m0-LIBERO-10k-step \
     --local-dir /path/to/ABot-m0-LIBERO-10k-step

Checkpoint 配置
~~~~~~~~~~~~~~~

ABot-M0 checkpoint 自带 ``config.yaml``。请检查 ``qwenvl.base_vlm`` 字段，并将其中的开发者本地路径替换为本机 Qwen3-VL backbone 路径：

.. code-block:: bash

   # 如果运行环境无法直接访问 Hugging Face，可启用镜像。
   # export HF_ENDPOINT=https://hf-mirror.com

   hf download StarVLA/Qwen3-VL-4B-Instruct-Action \
     --local-dir /path/to/Qwen3-VL-4B-Instruct-Action

.. code-block:: yaml

   qwenvl:
     base_vlm: /path/to/Qwen3-VL-4B-Instruct-Action

Qwen3-VL backbone 可从以下地址获取：
``https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action``。

离线加载 VGGT
~~~~~~~~~~~~~

ABot 当前默认使用以下方式初始化 VGGT：
``VGGT.from_pretrained("facebook/VGGT-1B")``

如果运行环境可以通过镜像访问 Hugging Face，可设置 ``HF_ENDPOINT``：

.. code-block:: bash

   export HF_ENDPOINT=https://hf-mirror.com

否则，请先离线下载 VGGT，并将其放入本地 Hugging Face cache 或本地目录：

.. code-block:: bash

   hf download facebook/VGGT-1B --local-dir /path/to/VGGT-1B

之后可采用以下方式之一：

* 放入本地 Hugging Face cache 目录，或
* 在 ABot 安装代码中将 VGGT 加载路径显式改为本地目录。

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

评测入口是 ``examples/embodiment/eval_embodied_agent.py``。两个 benchmark
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

   python examples/embodiment/eval_embodied_agent.py \
     --config-name libero_10_ppo_abot_m0 \
     actor.model.model_path=<path_to_abot_m0_ckpt> \
     rollout.model.model_path=<path_to_abot_m0_ckpt> \
     runner.only_eval=True \
     env.eval.total_num_envs=8 \
     env.eval.video_cfg.save_video=true \
     algorithm.eval_rollout_epoch=1 \
     runner.logger.experiment_name=abot_m0_libero10_eval

**LIBERO-Plus：**

.. code-block:: bash

   export LIBERO_TYPE=plus

   python examples/embodiment/eval_embodied_agent.py \
     --config-name libero_10_plus_ppo_abot_m0 \
     actor.model.model_path=<path_to_abot_m0_ckpt> \
     rollout.model.model_path=<path_to_abot_m0_ckpt> \
     runner.only_eval=True \
     env.eval.total_num_envs=8 \
     env.eval.video_cfg.save_video=true \
     algorithm.eval_rollout_epoch=1 \
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
