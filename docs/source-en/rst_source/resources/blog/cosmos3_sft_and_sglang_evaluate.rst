RLinf x SGLang: Integrating Cosmos3 from Fine-Tuning to Efficient Parallel Evaluation
=====================================================================================

Last updated: 09/21/2026.

Related reading: `RLinf x SGLang: Integrating Cosmos3 from Fine-Tuning to Efficient Parallel Evaluation <https://www.sglang.io/blog/rlinf-sglang-cosmos3>`_.

Abstract: From "supporting one more model" to "delivering a complete embodied pipeline."

As embodied-AI models keep evolving, the model itself is only one part of a complete development pipeline. For a new World Action Model (WAM), everything from model onboarding and training to inference and simulation-based evaluation requires infrastructure that can adapt quickly and run efficiently.

RLinf already supported world models such as DreamZero. Now, we have now integrated NVIDIA's Cosmos 3, with SGLang-Diffusion (hereafter "SGLang") as the inference backend, unifying SFT, inference serving, and simulation-based evaluation into one end-to-end pipeline. Cosmos3-Nano SFT on RLinf matches the official implementation's training accuracy, while strengthened SGLang multi-batch inference and pipelined parallel evaluation lift evaluation throughput to 3.33× that of a straightforward SGLang integration.

Using this Cosmos3-Nano integration as a case study, this post walks through the key design decisions behind the end-to-end path from fine-tuning to high-throughput parallel evaluation, and how RLinf's pipelined scheduling pushes evaluation efficiency further.

01 Cosmos3: A Step Up for RLinf's Embodied Stack
------------------------------------------------

Cosmos3 is NVIDIA's omni-modal model, capable of natively understanding and generating text, images, video, environmental audio, and actions. Its goal is to provide unified multimodal capabilities for modeling the physical world and generating actions.

Architecturally, Cosmos 3 adopts a Mixture-of-Transformers (MoT) dual-tower design, composed of an autoregressive Transformer and a diffusion Transformer.

1. The autoregressive Transformer handles multimodal understanding and reasoning.

2. The diffusion Transformer handles multimodal content generation.

The two towers share unified attention layers and a 3D mRoPE (3D Multimodal Rotary Position Embedding) positional representation, allowing the model to build a shared spatio-temporal representation across modalities and model spatio-temporal structure across modality boundaries.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/cosmos3-architecture.png
   :alt: Cosmos3 autoregressive and diffusion towers with shared multimodal attention
   :align: center
   :width: 100%

   （Source: Cosmos3 technical report, https://arxiv.org/pdf/2606.02800）

Cosmos 3 ships in three sizes — Edge, Nano, and Super. All training and evaluation described below uses Cosmos3-Nano.

Cosmos3-Nano is a 16B-parameter MoT model whose core follows the Qwen3-VL-8B architecture: 36 Transformer layers, hidden size 4096, 32 attention heads, and 8 KV heads.

Cosmos3-Nano is not trained entirely from scratch. The autoregressive Transformer reuses Qwen3-VL-8B's pretrained weights directly, while the diffusion Transformer is initialized from Qwen3-VL-8B weights and further optimized during Cosmos3 training.

The current technical setup for Cosmos3 in RLinf is as follows:

1. SFT training runs on RLinf's existing FSDP2 training backend.

2. Evaluation uses SGLang as the inference backend.

Cosmos3 is a fairly complex new multimodal/action model, and bringing it into an existing embodied-AI training framework is itself a representative engineering exercise. Choosing a model like this is a step up: RLinf has to support not a routine inference task, but a complete embodied-AI workflow spanning multimodal understanding, action generation, and simulation-based evaluation.

02 Model SFT: Lightweight Onboarding on Top of RLinf's Existing Framework
-------------------------------------------------------------------------

1. Lightweight adapter-based onboarding: decoupling the model implementation from the training framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cosmos3 is an OmniMoT model, and its architecture is already open-sourced in NVIDIA's cosmos-framework. For a new model that already has a complete official implementation, the biggest integration cost is not "getting it to run" — it is onboarding it into an existing training framework without ending up maintaining a second copy of the model implementation.

During integration, RLinf did not rewrite the Cosmos3 model architecture. Instead, it treats Cosmos3 as an external Hugging Face model and brings it in through an adapter layer.

Model construction and initialization remain the responsibility of Cosmos's own configuration system; RLinf only adds a thin wrapper around the model that maps Cosmos3's interfaces onto RLinf's.

Cosmos3 SFT runs on RLinf's existing FSDP2 training backend, so there is no need to build a dedicated training loop for Cosmos3. Checkpoint save/load, logging, and monitoring all reuse RLinf's existing mechanisms.

The core value of this design: it reuses the official Cosmos3 implementation as fully as possible without rewriting a separate copy of Cosmos3 model code in RLinf.

Decoupling model adaptation from training framework in this way also gives future models a cleaner, lower-effort integration path: a new model can reuse the existing training stack without any changes to RLinf's training backend.

2. Cosmos3-Nano SFT on RLinf, with training accuracy matching the official implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use NVIDIA's open-source LIBERO_LeRobot_v3 dataset, which contains multimodal trajectory data for robot manipulation tasks: RGB images from the primary and wrist-mounted camera views, robot state (8D), and the corresponding robot actions (7D). Actions are represented as 3D translation + 3D axis-angle rotation + gripper.

After loading LIBERO_LeRobot_v3, Cosmos3 converts and normalizes the action representation on the fly during data processing. The original 3D axis-angle observations are converted to a 6D rotation representation (Rot6D), the original 7D actions are converted to a 10D action representation, and the result is then normalized using quantile-based normalization.

Compared with axis-angle, Rot6D avoids the discontinuity around ±π, giving the action space a smoother, continuous representation. This helps the diffusion model learn the action distribution more stably and generate higher-quality robot actions.

.. note::

   the current SFT experiments use NVIDIA's official LIBERO_LeRobot_v3 dataset, whose videos are recorded at 10 FPS. If you switch to a different LIBERO dataset or rebuild your own, pay close attention to whether frame rate and related hyperparameters still match the current Cosmos3 configuration.

Results from 5,000 SFT iterations of Cosmos3 on RLinf are shown below (matching the official Cosmos3 implementation at https://github.com/NVIDIA/cosmos):

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/cosmos3-sft-training-loss.png
   :alt: Overall training loss of Cosmos3-Nano during 5,000 SFT iterations on LIBERO_LeRobot_v3
   :align: center
   :width: 100%

   Figure 1. Overall training loss of Cosmos3-Nano during 5,000 SFT iterations on LIBERO_LeRobot_v3

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/cosmos3-sft-action-flow-matching-loss.png
   :alt: Action flow-matching loss during Cosmos3-Nano SFT
   :align: center
   :width: 100%

   Figure 2. Action flow-matching loss during Cosmos3-Nano SFT

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/cosmos3-sft-vision-flow-matching-loss.png
   :alt: Vision flow-matching loss during Cosmos3-Nano SFT
   :align: center
   :width: 100%

   Figure 3. Vision flow-matching loss during Cosmos3-Nano SFT

After 5,000 SFT iterations on LIBERO_LeRobot_v3, the overall training loss of Cosmos3-Nano converges steadily. Taken together, the results show that the Cosmos3-Nano integration in RLinf learns the visual and action distributions in the LIBERO data effectively, with a stable training process throughout.

03 Evaluation: Decoupled Scheduling and Pipelined Execution
-----------------------------------------------------------

Once Cosmos3-Nano has been fine-tuned on LIBERO_LeRobot_v3, the next step is to evaluate the trained model in simulation to verify how well it actually generates robot actions.

1. Decoupling task scheduling from inference scheduling: letting RLinf and SGLang each do their own job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf currently supports SGLang as the inference backend for Cosmos3 evaluation, running as a standalone inference service that handles model inference and concurrent request processing. On the RLinf side, each compute component is managed by a Worker, which owns that component's lifecycle and its data exchange with other components.

There is one key design decision here: RLinf does not take over SGLang's internal inference scheduling.

SGLang already has mature inference optimizations, its own Router, and its own request scheduling. If RLinf were to impose additional lifecycle management or scheduling on top, it would risk interfering with SGLang's own scheduling and degrading inference performance.

RLinf therefore uses a non-intrusive SGLang Worker architecture:

RLinf: task orchestration and data flow

SGLang: inference and its own scheduling

The SGLang Worker starts the SGLang inference process during initialization. From then on, its main job is data adaptation and request forwarding between RLinf and SGLang; it stays out of SGLang's internal model inference and request scheduling.

When another Worker issues an inference request, the SGLang Worker first converts the input into the format SGLang expects, sends it straight to the SGLang service, and waits for the inference task to complete. Once inference finishes, the SGLang Worker retrieves the model output, converts it back, and passes the result to the next Worker. At no point does RLinf interfere with SGLang's internal model inference, request scheduling, or Router mechanism.

This keeps RLinf's task scheduling decoupled from SGLang's inference scheduling: RLinf owns the overall task flow, SGLang owns its own inference efficiency. The two systems integrate cleanly while SGLang's existing inference optimizations remain fully intact.

2. Overlapping CPU simulation with GPU inference to raise hardware utilization and evaluation throughput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulation and inference belong to the same task chain but consume different kinds of compute. During Cosmos3's LIBERO evaluation, the compute profiles of different Workers are markedly heterogeneous. The LIBERO simulator runs environment simulation mostly on CPU, while SGLang runs model inference on GPU. Executing the two strictly in sequence leaves visible idle gaps between the simulation and inference stages, so neither CPU nor GPU is fully utilized.

The question therefore shifts from "can the model run at all?" to:

How do we get different compute stages to genuinely work in parallel?

RLinf's answer is a pipelined execution mechanism in the evaluation flow. Given a batch of evaluation tasks, RLinf splits them into multiple batches and processes them across multiple pipelines, so that different batches execute in an interleaved fashion across Workers — overlapping LIBERO simulation with SGLang inference as much as possible.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-evaluation-pipeline.png
   :alt: Data flow between SglangEmbodiedWorker, SglangServer, and the LIBERO simulator
   :align: center
   :width: 100%

Take LIBERO-10 as an example. A full evaluation run on a single 8-GPU server covers 10 tasks × 50 demos = 500 episodes.

RLinf launches 128 parallel simulation environments to run the evaluation, with each environment responsible for 4 simulation tasks. When the task count does not divide evenly into the number of parallel environments, task padding keeps the parallel execution width constant. In the current configuration, the 128 simulation environments are distributed evenly across the 8 GPUs — 16 environments per GPU — so simulation tasks run in parallel. On top of that, RLinf can further split the 128 parallel environments into N pipelines, each handling 128 // N environments. Within a pipeline, tasks cycle through LIBERO simulation → SGLang inference → LIBERO action execution; across pipelines, execution is scheduled in a pipelined manner.

1. The LIBERO simulator first runs its environments in parallel and collects the current observations.

2. RLinf assembles the images and other observations from multiple environments into a batch and sends it to SGLang.

3. SGLang uses its multi-batch, high-concurrency inference capability to process many requests simultaneously and generate the corresponding next actions.

4. RLinf takes the inference results and returns each action to its LIBERO environment for execution; the environment then produces the next observation, which feeds the next inference round.

This loop repeats until every episode has been evaluated.

The evaluation process as a whole becomes a pipelined collaboration between parallel LIBERO simulation, RLinf pipeline scheduling, and high-concurrency SGLang inference. Environment simulation on the CPU and model inference on the GPU overlap in time, cutting the wait between compute stages and raising both hardware utilization and evaluation throughput.

3. Results: up to 3.33× throughput over the single-batch baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We optimized the current Cosmos3 simulation-evaluation workload by combining RLinf's pipelined scheduling with SGLang's high-concurrency inference. End-to-end results:

.. list-table:: 128 episodes, 32 simulation environments
   :header-rows: 1
   :widths: 40 30 30

   * - Observations per Request
     - ``pipeline_stage_num``
     - Relative Throughput
   * - 1
     - 4
     - 1.00×
   * - 2
     - 2
     - 1.66×
   * - 4
     - 1
     - 1.80×

.. list-table:: 500 episodes, 128 simulation environments
   :header-rows: 1
   :widths: 40 30 30

   * - Observations per Request
     - ``pipeline_stage_num``
     - Relative Throughput
   * - 1
     - 16
     - 1.00×
   * - 2
     - 8
     - 1.70×
   * - 4
     - 4
     - 2.67×
   * - 8
     - 2
     - 3.33×
   * - 16
     - 1
     - 2.33×

The first column is the number of observations packed into each request sent to SGLang, and pipeline_stage_num is the number of pipelines the work is split across; the product of the two is held constant. The results show that combining RLinf's pipelined scheduling with SGLang's high-concurrency inference significantly increases Cosmos3 simulation-evaluation throughput, reaching 3.33× the single-batch baseline on the full 500-episode evaluation.

04 RLinf × SGLang × Cosmos3: From "Supporting One More Model" to "A Complete Embodied Pipeline, End to End"
-----------------------------------------------------------------------------------------------------------

Integrating Cosmos3 was never just about RLinf "supporting one more model." In this work, Cosmos3 in RLinf became a complete pipeline spanning model onboarding, SFT training, SGLang inference, and simulation-based evaluation. What this demonstrates is not support for one particular model, but how RLinf can onboard continually evolving embodied-AI models in a lightweight way, integrate training, inference, and evaluation into one end-to-end flow, and then push overall efficiency further through system-level scheduling.

For embodied AI, the value of infrastructure likewise goes beyond "getting the model running." It lies in organizing the model, training, inference, simulation, and task execution into a single system that works together efficiently — which is the core direction of RLinf's ongoing work on infrastructure for embodied AI and agents.

To learn more, visit https://github.com/RLinf/RLinf — stars, follows, and feedback are all welcome.

Cosmos3 SFT docs: :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>`

Cosmos3 SGLang Eval docs: :doc:`Cosmos3 SGLang Eval <../../evaluations/guides/cosmos3_sglang>`

RLinf Team & SGLang Team
