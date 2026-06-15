# MAFSDPActor 阶段工作说明

## 一、工作目标

本阶段工作的核心目标，是让 `MAFSDPActor` 支持 `DynamicRolloutResult`，从而完成多智能体 FSDP Actor 路径从旧数据结构向新数据结构的迁移。

之所以要做这件事，是因为当前系统正在逐步从 `RolloutResult` 过渡到 `DynamicRolloutResult`。后者更适合动态 batch、多智能体分组和后续的训练流程扩展。如果 `MAFSDPActor` 仍然停留在旧结构上，就无法和新的 rollout 数据链路对齐，也会影响训练与推理的一致性。

## 二、完成了什么

### 1. 完成了 `MAFSDPActor` 向 `DynamicRolloutResult` 的适配

主要修改集中在：

- `rlinf/workers/actor/ma_fsdp_actor_worker.py`

这一部分工作的目的，是让 `MAFSDPActor` 能够真正接收、处理并返回 `DynamicRolloutResult`，而不是继续沿用 `RolloutResult` 的处理方式。

这样做的原因是，两者在数据组织方式、batch 合并拆分方式以及训练时所依赖的信息上都有差异。如果继续使用旧逻辑，虽然表面接口可能能接上，但在推理、优势计算、训练拆 batch 和结果回传时都会出现不一致，最终导致运行错误或结果错误。

### 2. 对齐了 `MAFSDPActor` 与已有实现的角色分工

开发过程中参考了：

- `rlinf/workers/actor/fsdp_actor_worker.py`
- `rlinf/workers/actor/ma_megatron_actor_worker.py`

这样做的原因是：

- `FSDPActor` 提供了单模型 FSDP Actor 的主流程参考
- `MAMegatronActor` 提供了多智能体场景下已经完成的适配参考

因此本次工作本质上是在两者之间“对齐能力”：既保持 FSDP 路径的训练与推理框架，又补上多智能体和动态 rollout 结果所需要的处理方式。

### 3. 清理了 `ma_fsdp_actor_worker.py` 中重复和冲突的推理逻辑

在这个文件里原先存在两套 `run_inference` 相关逻辑，容易造成行为不一致，也不利于后续维护。

本次已经将推理路径收敛为一套明确逻辑。

这样做的原因是，如果同一职责存在多套实现，一方面容易让后续调试时无法判断真正生效的是哪一套，另一方面也容易在迁移到 `DynamicRolloutResult` 后出现一处已改、一处未改的问题。

## 三、配套完成的工作

### 4. 调整了 SearchR1 的训练配置与入口，使其切换到 `MAFSDPActor`

涉及文件：

- `examples/agent/searchr1/config/train_qwen2.5.yaml`
- `examples/agent/searchr1/train.py`

这部分工作的目的，是让 SearchR1 训练链路能够真正使用新的多智能体 FSDP Actor，而不是继续绑定 Megatron 路径。

同时，也按你的要求更新了模型路径和数据集路径。

这样做的原因是，仅仅实现 `MAFSDPActor` 本身还不够，如果训练入口和配置层没有切换过来，实际运行时仍然不会走到新的实现，前面的适配工作就无法生效。

### 5. 清理了部分 `torch.cuda` 的直接写法，改为按平台自适应

重点检查和处理了：

- `rlinf/hybrid_engines/sglang/common/sgl_scheduler.py`
- `rlinf/hybrid_engines/sglang/common/` 下的相关文件
- `rlinf/utils/data_iter_utils.py`
- `rlinf/data/io_struct.py`

这样做的原因是，当前运行环境不是标准 CUDA 路径，代码中如果直接写死 `torch.cuda`，在 NPU 或其他平台上会直接报错，特别是在动态 batch、scheduler 和 rollout 处理这些基础环节上，这类问题会非常早暴露出来。

因此这里的调整，本质上是在为 `MAFSDPActor` 所依赖的新链路扫清平台兼容障碍。

### 6. 移除了 `mp_wrapper` 相关逻辑

你已经确认过 `mp_wrapper` 是当前问题链路中的干扰项，因此本次已经将这个目录及相关引用一并移除。

这样做的原因是，这部分逻辑一方面已经不适合当前路径，另一方面还引入了初始化阶段的问题，继续保留只会增加排查复杂度。去掉以后，FSDP 模型和优化器初始化链路会更直接，也更容易判断后续问题究竟来自哪里。

## 四、排查并修复过的问题

### 7. 修复了动态 batch 场景下对 CUDA 的硬编码问题

你提供过一条典型报错，说明某些动态 batch 逻辑里直接把张量建在 `cuda` 上，导致在当前环境中运行失败。

这部分已经改为自适应选择设备。

这样做的原因是，动态 batch 是 `DynamicRolloutResult` 路径上的基础能力，如果这里仍然依赖固定 CUDA 设备，后面的多智能体训练和推理都无法正常工作。

### 8. 修复了 FSDP 参考策略权重切换路径的问题

你后面提供的报错里，`MAFSDPActor` 在计算参考策略相关结果时出现了大量参数 shape mismatch。

这个问题的根本原因，不是模型文件本身错误，而是 FSDP 场景下“参考策略权重切换”的方式不对。原有做法适合普通模型，但不适合 FSDP 这种带分片状态的模型。

因此本次已经把这条链路改成 FSDP 兼容的方式。

这样做的原因是，如果这里不改，`MAFSDPActor` 即使前面的数据结构迁移都完成了，也会在参考策略相关推理阶段继续失败，无法完成完整训练闭环。

### 9. 处理了 DTensor 与 CPU 状态拷贝相关的兼容问题

此前还出现过状态字典转到 CPU 时的兼容性问题，这类问题会影响参考策略保存、权重切换以及某些离线状态管理逻辑。

这部分已经做了兼容处理。

这样做的原因是，在 FSDP 场景下，模型状态并不总是普通张量。如果不处理这类兼容问题，一些本来看起来只是“保存一份状态”或者“临时切一次权重”的操作，也可能在运行时失败。

## 五、为什么这些工作必须一起做

这次开发表面上看是“实现 `MAFSDPActor`”，但实际上它不是单点修改，而是一条完整链路的打通，至少包含下面几层：

- Actor 本身要支持 `DynamicRolloutResult`
- 训练入口要真的切到 `MAFSDPActor`
- 动态 batch 和 rollout 相关基础能力要支持当前平台
- FSDP 特有的参考策略切换逻辑要可用

如果只改其中一层，系统仍然会在其他环节报错。因此本阶段的工作本质上是在逐步打通一整条可运行路径，而不是只补一个类。

## 六、当前阶段结论

到目前为止，围绕 `MAFSDPActor` 的主体迁移工作已经基本完成，相关配套修改也已经补齐，包括：

- 多智能体 FSDP Actor 对 `DynamicRolloutResult` 的支持
- SearchR1 训练入口和配置切换
- 动态 batch 与平台适配问题清理
- `mp_wrapper` 移除
- FSDP 参考策略切换问题修复

由于按要求没有在本地运行代码，所以最终验证仍然需要你上传到服务器后完成。后续如果还有新的运行时报错，可以继续基于服务端日志逐步收敛。

