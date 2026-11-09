Asynchronous Search-R1 Training
================================

Run Search-R1 with the rollout and the training overlapped, so that prompts
keep flowing into the rollout engines while the actor trains on an earlier
weight version. This page uses
``examples/agent/searchr1/config/train_glm4.7_flash_2node_async.yaml`` on two
8-GPU nodes with the actor and the rollout placed on separate nodes. Set up
the environment and the Qdrant wiki retrieval server first, as described in
:doc:`index`; the model-specific configuration (data format, tokenizer,
parallelism of the actor) is the same as in :doc:`glm4.7_flash`. For how the
asynchronous mode works and how to size its scheduling limits, read
:doc:`Asynchronous Agentic RL </rst_source/guides/async_agentic>`.

Overview
--------

Use this recipe to train GLM-4.7-Flash asynchronously on two 8-GPU nodes:
the Megatron actor occupies node 0 and the SGLang server rollout plus the
reward worker occupy node 1, and generation runs continuously while training
consumes the data of an earlier weight version.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      GLM-4.7-Flash (MoE, about 3B active parameters)

   .. grid-item-card:: Algorithm
      :text-align: center

      Multi-turn RL with search-tool calls, asynchronous rollout

   .. grid-item-card:: Tools
      :text-align: center

      Qdrant local wiki server

   .. grid-item-card:: Hardware
      :text-align: center

      Reference run on 2×8 A100-80G

| **You'll do:** set up the Ray cluster → fill in three paths → launch →
  watch ``rollout/buffer/*``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` ·
  :doc:`multi-node guide </rst_source/guides/multi_node>` · the retrieval
  server from :doc:`index`.

Requirements
------------

Asynchronous training relies on these four pieces working together:

- A disaggregated placement: the actor and the rollout run on different GPUs
  (in this config, on different nodes), so that training does not contend
  with generation for the same devices.
- A Megatron actor: the actor worker implements the async training loop.
- ``rollout.sglang.backend_type: server``: the SGLang server backend, which
  supports the weight sync and pause protocol the async runner uses.
- ``agentloop.task_name``: the task name that keys the rollout buffer's
  per-task state and is stamped on every rollout request. Async training
  fails at startup when it is unset.

The reward is computed inside the agent loop (the Search-R1 rule-based
reward parses the trajectory and the retrieved documents), so no separate
reward-model inference step runs between rollout and training.

Run It
------

Set up the Ray cluster as described in the
:doc:`multi-node guide </rst_source/guides/multi_node>`: export a unique
``RLINF_NODE_RANK`` on every node (0 on the head) and activate the RLinf
environment before ``ray start``. Start the Qdrant retrieval server on
node 0 before training; the search tool worker runs there, so
``tools.search.server_addr`` only needs to be reachable from node 0.

Fill in the data path, the model path, and the retrieval server address:

.. code-block:: yaml

   data:
     train_data_paths: ["/path/to/data/searchr1_train_glm.jsonl"]
   rollout:
     model:
       model_path: /path/to/model/GLM-4.7-Flash
   tools:
     search:
       server_addr: <server_ip>:8000

Then start the training from the head node:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_glm4.7_flash_2node_async

Configuration
-------------

Beyond the synchronous recipe, these keys control the asynchronous mode:

.. list-table::
   :header-rows: 1
   :widths: 40 18 42

   * - Key
     - Default
     - What it does
   * - ``runner.async_mode``
     - ``False``
     - Switches the runner to the async agent runner, which overlaps rollout
       with training.
   * - ``agentloop.task_name``
     - required
     - Keys the rollout buffer's per-task state; stamped on every rollout
       request.
   * - ``rollout.pause_mode``
     - ``retract``
     - How generation pauses for a weight sync: ``retract`` returns in-flight
       requests to the queue and keeps the tokens already generated;
       ``in_place`` pauses the engine's event loop while the running requests
       keep their KV cache and resume decoding with the new weights;
       ``drain`` waits for all in-flight requests to finish before the sync.
   * - ``rollout_buffer.max_inflight_groups``
     - derived
     - How many query groups may be issued but not returned. Derived as
       ``ceil(max_running_requests x instances x 1.2 / group_size)`` when
       unset (77 with this config).
   * - ``rollout_buffer.max_buffered_groups``
     - ``2 x rollout_batch_size``
     - Limit on in-flight plus buffered groups. Bounds both buffer memory
       and how stale training data can get. Must be at least
       ``rollout_batch_size`` (config validation rejects smaller values);
       an ``agentloop.max_actor_accept_staleness`` below
       ``ceil(max_buffered_groups / rollout_batch_size)`` triggers a
       warning at startup.
   * - ``rollout_buffer.max_dropped_groups_per_step``
     - ``10 x rollout_batch_size``
     - Safety limit on dropped groups between two hand-offs; the run fails when
       it is exceeded.
   * - ``agentloop.max_actor_accept_staleness``
     - unlimited
     - Groups staler than this are dropped before training; the drop frees
       admission capacity for other prompts, and the dropped prompt itself
       comes around again in a later epoch.
   * - ``down_sampling.down_sampling_config.reject_equal_reward``
     - ``False``
     - Drops groups whose per-trajectory rewards are all equal (no gradient
       signal for a within-group baseline).

How to size the two scheduling limits depends on whether generation or
training is the bottleneck; see :doc:`Asynchronous Agentic RL
</rst_source/guides/async_agentic>` for the mechanism and the measured
evidence.

Metrics to Watch
----------------

These TensorBoard series tell you how the async pipeline is doing:

- ``rollout/buffer/inflight_groups``: how many query groups are issued but
  not yet returned. Pinned at the derived limit means the rollout engines are
  saturated.
- ``rollout/buffer/size``: how many finished groups wait in the buffer.
  Compare it against ``max_buffered_groups``.
- ``rollout/buffer/admission_wait_inflight_limit`` and
  ``rollout/buffer/admission_wait_buffer_full``: the fraction of the step
  window during which admission was blocked by each limit. Together they tell
  you which limit is binding.
- ``rollout/data_staleness/mean``: for the groups of the step, the actor's
  version at step start minus the earliest version a group was generated
  with — averaged. The companion series
  ``rollout/version_span/mean`` (generation-side spread) and
  ``rollout/queue_steps/mean`` (time spent waiting in the buffer) decompose
  it.
- ``rollout/buffer/dropped_groups``: how many groups were dropped per step
  (by the staleness limit or the equal-reward filter). Sustained high values
  waste generation compute.

Visualization and Results
-------------------------

The reference run trains GLM-4.7-Flash on 2×8 A100-80G with
``rollout_batch_size`` 128 × ``group_size`` 4 and ``pause_mode: retract``,
for 20 steps (statistics over steps 2–19, after the first-step warm-up).
Throughput is measured per step as ``total_num_sequence x
response_length / time/step``.

.. list-table::
   :header-rows: 1
   :widths: 30 16 16 16 11 11

   * - Run
     - Mean step time
     - Throughput
     - Mean data staleness
     - reward range
     - ``ratio_abs`` range
   * - Synchronous ``AgentRunner``
     - 278.6 s
     - 1159 tok/s
     - —
     - 0.25–0.47
     - 0.052–0.062
   * - Asynchronous
     - 139.0 s
     - 2024 tok/s
     - 1.3–1.4
     - 0.26–0.42
     - 0.053–0.063

Training quality holds: the reward range and the rollout-training logprob
gap (``ratio_abs``) match the synchronous run, with the step time halved
(278.6 s to 139.0 s) and throughput up 1.75× (1159 to 2024 tok/s). The
in-flight group count stays pinned at the derived limit of 77, with 98–100%
of each step window blocked by the in-flight limit — the rollout engines are
saturated the whole time. The buffer holds 72–118 groups against its limit of
256. Responses grow from about 180 to about 380 tokens over the 20 steps,
so the step time drifts up slowly while the token throughput holds.

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_step_time.png" style="width: 100%;"/>
       <p><em>Time per step, synchronous vs asynchronous</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_reward_walltime.png" style="width: 100%;"/>
       <p><em>Trajectory reward against wall-clock time</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_reward_step.png" style="width: 100%;"/>
       <p><em>Trajectory reward per step</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_ratio_abs.png" style="width: 100%;"/>
       <p><em>Rollout-training logprob gap (ratio_abs)</em></p>
     </div>
   </div>
   <p style="text-align: center;"><em>Asynchronous Search-R1 on GLM-4.7-Flash</em></p>

The x-axis of these plots counts parameter updates: each rollout step
performs four of them (``algorithm.n_minibatches: 4``), the same convention
as in :doc:`glm4.7_flash`.

Tuning the Limits
-----------------

Two measured cases show how each limit behaves when a different side is
the bottleneck. The reasoning behind the numbers is in
:doc:`Asynchronous Agentic RL </rst_source/guides/async_agentic>`; this
section keeps the evidence.

When generation is the bottleneck (GLM, this config), raising the in-flight
limit does not buy throughput. Raising ``max_inflight_groups`` from the
derived 77 to 160 moves the running requests per DP rank from about 75 to
128 (with another 33–78 queued), while the throughput stays flat and the
mean staleness rises from about 1.3 to about 2.0: the GPUs are already busy
at 75 running requests.

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_cap_step_time.png" style="width: 100%;"/>
       <p><em>Time per step, in-flight limit 77 vs 160</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_cap_staleness.png" style="width: 100%;"/>
       <p><em>Mean data staleness, in-flight limit 77 vs 160</em></p>
     </div>
   </div>

When training is the bottleneck (Qwen2.5-1.5B-Instruct, 8 SGLang instances
at TP1, ``rollout_batch_size`` 64 × ``group_size`` 4), the buffer limit
decides whether the two sides overlap. At ``max_buffered_groups`` =
1 × ``rollout_batch_size``, the next batch can only start generating after
the current one is handed off, so generation and training serialize and the
median step time is about 11 s. At 2 ×, the median step time is 3.94 s; the
buffer settles at 54 and 108 groups respectively, both within the limit.

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_async_buffer_step_time.png" style="width: 100%;"/>
       <p><em>Time per step, buffer limit 1×B vs 2×B</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_async_buffer_size.png" style="width: 100%;"/>
       <p><em>Buffer size, limit 1×B vs 2×B</em></p>
     </div>
   </div>
   <p style="text-align: center;"><em>In-flight and buffer limit experiments</em></p>

Resuming
--------

The rollout buffer worker saves the dataloader position into the checkpoint
(``data/data.pt``). On resume, the prompts that were already issued before
the save are skipped, and their count is logged at every save; the measured
count is 192 on the Qwen default config (128 in-flight plus buffered
groups, plus one prefetched dataloader batch of 64).

Limitations
-----------

- Asynchronous training is implemented for the Megatron actor with the
  multi-turn agent loop used by Search-R1.
- The rollout logprob is used directly as the PPO old logprob
  (``algorithm.recompute_logprobs: False``).
- With ``reject_equal_reward`` enabled the drop rate can be high; check
  ``rollout/buffer/dropped_groups`` before committing to a long run.
