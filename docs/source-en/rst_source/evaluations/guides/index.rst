Benchmarks
==========

Pick a benchmark below for its end-to-end installation, checkpoint download, launch command, and troubleshooting guide.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: BEHAVIOR-1K
      :link: behavior
      :link-type: doc

      50 household tasks on OmniGibson and Isaac Sim 4.5 with OpenPI π₀.₅.

   .. grid-item-card:: LIBERO
      :link: libero
      :link-type: doc

      Spatial, Object, Goal, 10, and 90 suites, plus LIBERO-PRO and LIBERO-PLUS.

   .. grid-item-card:: ManiSkill OOD
      :link: maniskill_ood
      :link-type: doc

      Batch evaluation across 13 out-of-distribution scenes and in-distribution splits.

   .. grid-item-card:: PolaRiS
      :link: polaris
      :link-type: doc

      DROID tabletop manipulation on Isaac Sim 5.1.0 with OpenPI π₀ / π₀.₅.

   .. grid-item-card:: RealWorld
      :link: realworld
      :link-type: doc

      Franka PnP, custom ``FrankaEnv-v1`` tasks, and Dual Franka RTC real-robot evaluation.

   .. grid-item-card:: RoboLab
      :link: robolab
      :link-type: doc

      Single-task and full 120-task DROID joint-position evaluation with π₀.₅.

   .. grid-item-card:: RoboTwin
      :link: robotwin
      :link-type: doc

      Bimanual manipulation tasks with OpenVLA-OFT, OpenPI, and LingbotVLA.

   .. grid-item-card:: SIMPLE
      :link: simple
      :link-type: doc

      Eight G1 whole-body humanoid Teleop tasks evaluated with Psi0 across 3 DR levels.

   .. grid-item-card:: SuperMimic
      :link: supermimic
      :link-type: doc

      Unitree G1 motion tracking and standing push recovery.

SGLang Serving Guides
---------------------

Dedicated guides for world-action models served through RLinf's SGLang embodied backend on LIBERO:

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Cosmos3 (SGLang)
      :link: cosmos3_sglang
      :link-type: doc

      LIBERO-Spatial evaluation with SGLang diffusers serving for Cosmos3.

   .. grid-item-card:: DreamZero (SGLang)
      :link: dreamzero_sglang
      :link-type: doc

      LIBERO-Spatial evaluation with SGLang serving for DreamZero WAN2.2-5B.

.. note::

   Configs for GenieSim, RoboCasa365, and RoboDojo also live under ``evaluations/`` (see :doc:`../index`). For simulators without a dedicated ``evaluations/`` folder such as IsaacLab or MetaWorld, :doc:`config fallback <../reference/cli>` loads eval-ready YAMLs from ``examples/embodiment/config/`` that define ``rollout.model`` (with ``model_type`` and ``num_action_chunks``); to evaluate a training-only config from :doc:`../../examples/simulators_index`, adapt it per :doc:`../reference/configuration`.

.. toctree::
   :hidden:
   :maxdepth: 2

   BEHAVIOR-1K <behavior>
   LIBERO <libero>
   ManiSkill OOD <maniskill_ood>
   PolaRiS <polaris>
   RealWorld <realworld>
   RoboLab <robolab>
   RoboTwin <robotwin>
   SIMPLE <simple>
   SuperMimic <supermimic>
   Cosmos3 (SGLang) <cosmos3_sglang>
   DreamZero (SGLang) <dreamzero_sglang>
