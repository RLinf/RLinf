MUSA 上的 ManiSkill 需要使用包含修改版 SAPIEN 与匹配 ManiSkill 包的厂商镜像。在仓库根目录启动：

.. code-block:: bash

   export MUSA_IMAGE=registry.mthreads.com/lgpublic/rlinf:rlinf0.2-maniskill_libero_openpi-0428-jingdong
   docker run -it --rm --runtime=mthreads \
      --ipc=host --shm-size=100g \
      -e MTHREADS_VISIBLE_DEVICES=all \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      "$MUSA_IMAGE" bash
   download_assets --assets maniskill

.. warning::

   公开的 ``sapien`` 与 ManiSkill ``v3.0.0b22`` 无法组成可用的 MUSA 模拟器环境。请保留镜像中的厂商包；添加其他模型环境时使用 ``--env libero``，避免 ``install.sh`` 替换这些模拟器包。
