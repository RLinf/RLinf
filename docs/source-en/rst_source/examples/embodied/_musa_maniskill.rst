ManiSkill on MUSA requires the vendor image that supplies the modified SAPIEN
and matching ManiSkill packages. Start it from the repository root:

.. code-block:: bash

   export MUSA_IMAGE=registry.mthreads.com/lgpublic/rlinf:rlinf0.2-maniskill_libero_openpi-0428-jingdong
   docker run -it --rm --runtime=mthreads \
      --ipc=host --shm-size=100g \
      -e MTHREADS_VISIBLE_DEVICES=all \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      "$MUSA_IMAGE" bash
   download_assets --assets maniskill

.. warning::

   The public ``sapien`` and ManiSkill ``v3.0.0b22`` packages do not form a
   working MUSA simulator stack. Keep the packages from this image. When adding
   another model environment, install it with ``--env libero`` so
   ``install.sh`` does not replace the vendor simulator packages.
