ManiSkill can run PhysX simulation on CPU independently of the accelerator that
hosts the model. Apply these settings to both ``env.train`` and ``env.eval`` in
the selected ManiSkill config:

.. code-block:: yaml

   env:
     train:
       total_num_envs: 2
       init_params:
         sim_backend: cpu
         render_backend: "pci:0000:00:00.0"
     eval:
       total_num_envs: 2
       init_params:
         sim_backend: cpu
         render_backend: "pci:0000:00:00.0"

``sim_backend: cpu`` moves PhysX off CUDA; the VLA still runs on the selected
accelerator. ``render_backend`` selects the SAPIEN Vulkan renderer by its full
PCI address. RLinf preserves a ``pci:<domain>:<bus>:<slot>.<function>`` value
before creating the environment; replace the sample address if the renderer
inside your container uses another address.

.. warning::

   The CPU simulation backend cannot vectorize several environments inside one
   process. Set each ``total_num_envs`` to the number of env worker ranks, so
   each worker owns one environment.
