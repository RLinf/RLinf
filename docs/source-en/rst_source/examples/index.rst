Embodied AI
===========

Choose an entry point by benchmark, model family, or training paradigm:

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Simulators
      :link: simulators_index
      :link-type: doc

      Start from a benchmark — LIBERO, ManiSkill, RoboTwin, BEHAVIOR-1K, IsaacLab, and more.

   .. grid-item-card:: Embodied Models
      :link: vla_wam_index
      :link-type: doc

      RL-fine-tune VLA and WAM families — π₀ / π₀.₅, OpenVLA-OFT, GR00T, StarVLA, LingBot-VLA, ABot-M0, and more.

   .. grid-item-card:: SFT & Imitation
      :link: sft_index
      :link-type: doc

      Supervised fine-tuning and imitation recipes — OpenPI, DreamZero, FastWAM, Cosmos3, Qwen-VL, and DAgger.

   .. grid-item-card:: RL Recipes
      :link: methods_index
      :link-type: doc

      Algorithm-focused recipes — SAC-Flow, DSRL, RL Token, Sim-Real Co-Training, OPD, RECAP, STEAM, and IQL.

   .. grid-item-card:: World Models
      :link: world_model_index
      :link-type: doc

      Closed-loop VLA RL using learned world models as simulators — OpenSora, Wan, and BWM.

Model × Simulator Quick Matrix
------------------------------

Find the right recipe from either a model family or a benchmark.

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 10 10 10 10 10 10

   * - Model
     - :doc:`BEHAVIOR-1K <embodied/behavior>`
     - :doc:`CALVIN <embodied/calvin>`
     - :doc:`LIBERO <embodied/libero>`
     - :doc:`ManiSkill3 <embodied/maniskill>`
     - :doc:`MetaWorld <embodied/metaworld>`
     - :doc:`RoboCasa <embodied/robocasa>`
     - :doc:`RoboTwin <embodied/robotwin>`
     - :doc:`Real-World <real_world_index>`
   * - :doc:`Dexbotic <embodied/dexbotic>`
     - —
     - —
     - ✓
     - —
     - —
     - —
     - —
     - —
   * - :doc:`GR00T <embodied/gr00t>`
     - —
     - —
     - ✓
     - ✓
     - —
     - —
     - —
     - —
   * - :doc:`MLP <embodied/mlp>` / :doc:`SAC <embodied/sac_flow>`
     - —
     - —
     - ✓
     - ✓
     - —
     - —
     - —
     - ✓
   * - :doc:`OpenPI π₀ / π₀.₅ <embodied/pi0>`
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - :doc:`OpenVLA <embodied/maniskill>`
     - —
     - —
     - ✓
     - ✓
     - —
     - —
     - —
     - —
   * - :doc:`OpenVLA-OFT <embodied/openvla_oft>`
     - ✓
     - —
     - ✓
     - ✓
     - ✓
     - —
     - ✓
     - —
   * - :doc:`StarVLA <embodied/starvla>`
     - —
     - —
     - ✓
     - —
     - —
     - —
     - —
     - —

Looking for other tracks? See :doc:`Real-World Robots <real_world_index>`, :doc:`Evaluation <../evaluations/index>`, :doc:`Agentic & Reasoning <agentic/index>`, and :doc:`Image & Video Generation <video_models>`.
