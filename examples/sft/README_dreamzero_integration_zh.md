# DreamZero 接入 RLinf SFT 说明

本文档说明这次为了将 DreamZero 的 SFT 训练流程接入 RLinf，我具体做了哪些代码改动，以及每一处改动背后的设计原因。

## 1. 目标

这次接入的目标不是把 DreamZero 原始训练系统完整搬进 RLinf，而是先打通一条最小可训练链路：

- 使用 RLinf 的 SFT runner 和 FSDP worker 启动训练
- 复用 DreamZero 原有的 Hydra 训练配置、数据集定义和 data collator
- 复用 DreamZero 原有的 `VLA` 模型结构和 checkpoint
- 训练完成后，保存出 DreamZero 推理仍能识别的关键文件

换句话说，这次的核心目标是：

`RLinf 负责调度和分布式训练`

`DreamZero 继续负责模型结构、数据定义和训练语义`

## 2. 接入前的主要问题

在改动之前，DreamZero 虽然已经在 RLinf 代码里“出现”了，但实际上还不能真正跑通 SFT。主要有四个断点。

### 2.1 `FSDPVlaSftWorker` 没有真的支持 DreamZero dataloader

文件：

- `RLinf/rlinf/workers/sft/fsdp_vla_sft_worker.py`

问题：

- `build_dataloader()` 只支持 `OPENPI` 和 `LINGBOTVLA`
- 虽然 `get_train_model_output()` 里已经有 `DREAMZERO_SFT` 分支，但数据构建层没接上

这意味着：

- DreamZero 模型即使能被识别
- 训练时也没有办法从 RLinf 的 `data.train_data_paths` 构建出 DreamZero 所需 batch

### 2.2 RLinf 的训练模型接口与 DreamZero wrapper 不一致

文件：

- `RLinf/rlinf/models/embodiment/dreamzero/dreamzero_action_model.py`

问题：

- 这个类本质上是一个推理/评估包装器
- 里面真正可训练的是 `self.model`，也就是 DreamZero 的底层 `VLA`
- 但 RLinf 的 FSDP manager 需要的是一个可直接 wrap 的 `nn.Module`

如果继续直接拿这个 wrapper 当训练对象，会有几个风险：

- FSDP wrap 行为不稳定
- forward 接口和 RLinf SFT worker 的调用方式不完全一致
- checkpoint 保存/恢复的语义不清晰

### 2.3 DreamZero 原训练严重依赖 Hydra 配置实例化

文件：

- `groot/vla/experiment/base.py`
- `groot/vla/configs/data/dreamzero/*.yaml`
- `groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain.yaml`

问题：

- DreamZero 训练不是“给一个 data root 就完了”
- 它的数据集、transform、collator 都是通过 Hydra config 动态实例化出来的

这意味着：

- 不能简单像 OpenPI 一样只靠一个轻量 dataconfig 去拼装数据
- 必须复用 DreamZero 自己的训练配置，否则 batch 字段和预处理逻辑很容易错

### 2.4 RLinf 默认 checkpoint 不足以直接兼容 DreamZero 推理侧

问题：

- DreamZero 推理时不只读权重
- 还依赖 `experiment_cfg/conf.yaml`、`experiment_cfg/metadata.json` 和 `config.json`

如果只保存 FSDP 权重：

- 训练可以结束
- 但后续 DreamZero 的 eval / policy wrapper 很可能加载不起来

## 3. 这次做了哪些改动

---

## 3.1 新增 DreamZero SFT builder

文件：

- [RLinf/rlinf/models/embodiment/dreamzero/sft_builder.py](/Users/wwxq/dreamzero/RLinf/rlinf/models/embodiment/dreamzero/sft_builder.py)

这是这次接入最核心的新文件，主要负责三件事。

### 改动 1：加载 DreamZero 训练配置

新增函数：

- `load_dreamzero_train_cfg()`

作用：

- 从 `actor.model.train_cfg_path` 读取 DreamZero 的原始训练配置
- 如果没有显式指定，就默认读 `model_path/experiment_cfg/conf.yaml`
- 支持用 RLinf 的 `data.train_data_paths` 覆盖 DreamZero config 里的 `*_data_root`

为什么这么做：

- RLinf 的配置风格是“训练入口统一，运行时传 override”
- DreamZero 的配置风格是“训练细节写在 Hydra config 里”
- 这个函数就是两套配置系统之间的桥接层

设计原因：

- 不重写 DreamZero 的数据语义
- 不把 DreamZero 大量配置字段平铺进 RLinf
- 只在必要的位置做最薄的一层 override

### 改动 2：构建可训练的 DreamZero 模型

新增函数：

- `build_dreamzero_sft_model()`

作用：

- 直接实例化 DreamZero 底层 `VLA` 模型
- 从 checkpoint 加载权重
- 如有需要，在权重加载后注入 LoRA adapter

为什么这么做：

- RLinf 的 FSDP manager 需要拿到真正的 `nn.Module`
- 与其让 FSDP 去包一层评估 wrapper，不如直接拿 DreamZero 底层训练模型

设计原因：

- 训练和推理职责分离
- 训练路径尽量贴近 DreamZero 原始实现
- 减少 wrapper 带来的接口不一致问题

### 改动 3：构建 DreamZero dataloader

新增函数：

- `build_dreamzero_sft_dataloader()`

作用：

- 复用 DreamZero 原始 Hydra config 中的 `train_dataset`
- 复用 DreamZero 原始 `data_collator`
- 按 RLinf 的 `micro_batch_size`、`num_workers` 等参数构建 `DataLoader`
- 返回 `data_config`，把训练配置、metadata、模型 config 一并带回 worker

为什么这么做：

- DreamZero 的 batch 字段是定制的
- 尤其语言、视频、多视角和 action 的处理都依赖原始 collator
- 直接复用原始 collator 比自己重新拼 batch 稳得多

设计原因：

- 最大限度避免 batch schema 漂移
- 保持 DreamZero 原有训练数据处理逻辑不变

---

## 3.2 修改 `FSDPVlaSftWorker`，接入 DreamZero 训练分支

文件：

- [RLinf/rlinf/workers/sft/fsdp_vla_sft_worker.py](/Users/wwxq/dreamzero/RLinf/rlinf/workers/sft/fsdp_vla_sft_worker.py)

这是这次最重要的接入点。

### 改动 1：新增 DreamZero 专属 `model_provider_func()`

作用：

- 当 `model_type` 是 `dreamzero` 且任务是 SFT 时
- 不再走通用 `get_model()` 路径
- 而是直接调用 `build_dreamzero_sft_model()`

为什么这么做：

- 避免训练时错误地拿到推理 wrapper
- 明确 DreamZero SFT 的训练对象就是底层 `VLA`

### 改动 2：在 `build_dataloader()` 中新增 DreamZero 分支

作用：

- 当 `model_type` 是 `DREAMZERO_SFT` 时
- 调用 `build_dreamzero_sft_dataloader()`

为什么这么做：

- 把 DreamZero 的数据链路真正接到 RLinf worker 上

### 改动 3：修正 `get_train_model_output()` 的 batch 使用逻辑

作用：

- 原来 LingBot 和 DreamZero 分支里虽然函数签名收了 `batch`
- 但内部又自己调用了一次 `next(self.data_iter)`
- 等于传进来的 batch 被忽略了

我改成了：

- 直接使用函数参数 `batch`

为什么这么做：

- 避免重复读数据
- 避免 batch 跳步
- 让 worker 的控制流更符合 RLinf 框架本身的设计

这是一个顺手修掉的小 bug。

### 改动 4：新增 data state 的保存与恢复

新增方法：

- `_save_data_state()`
- `_load_data_state()`
- `save_checkpoint()`
- `load_checkpoint()`

作用：

- 在 checkpoint 中保存 `data_epoch` 和 `data_iter_offset`
- resume 时尽量恢复到原来的数据位置

为什么这么做：

- RLinf 的 SFT 训练是自己控制 dataloader 迭代的
- 不像 HuggingFace Trainer 那样自动帮你处理数据跳过逻辑
- 不补这层的话，恢复训练时数据进度会漂

### 改动 5：新增 DreamZero 专属训练产物保存

新增方法：

- `_save_dreamzero_artifacts()`

保存内容：

- `experiment_cfg/conf.yaml`
- `experiment_cfg/metadata.json`
- `config.json`

为什么这么做：

- DreamZero 的推理和后处理依赖这些文件
- 如果只存 FSDP 权重，训练结果很难被 DreamZero 生态继续使用

设计原因：

- 让 RLinf 训练出来的 checkpoint 尽量保持 DreamZero 兼容性

---

## 3.3 扩展 DreamZero checkpoint 加载逻辑

文件：

- [RLinf/rlinf/models/embodiment/dreamzero/dreamzero_action_model.py](/Users/wwxq/dreamzero/RLinf/rlinf/models/embodiment/dreamzero/dreamzero_action_model.py)

改动：

- `_load_model_with_config()` 现在除了支持 `model.safetensors` 外
- 也支持读取：
  - `model_state_dict/full_weights.pt`
  - `actor/model_state_dict/full_weights.pt`

为什么这么做：

- RLinf FSDP 保存出来的完整权重文件通常是 `full_weights.pt`
- 如果 DreamZero wrapper 不认识这个格式
- 那么 RLinf 训练完的 checkpoint 后续仍然不好复用

设计原因：

- 让 DreamZero 的现有推理代码更容易读取 RLinf 训练产物

---

## 3.4 新增 DreamZero SFT 配置模板

文件：

- [RLinf/examples/sft/config/dreamzero_sft_droid.yaml](/Users/wwxq/dreamzero/RLinf/examples/sft/config/dreamzero_sft_droid.yaml)

作用：

- 提供一份最小可改的 DreamZero SFT 配置模板
- 你只需要填这几个路径：
  - `data.train_data_paths`
  - `actor.model.model_path`
  - `actor.model.pretrained_model_path`
  - `actor.model.train_cfg_path`
  - `actor.model.data_root_key`

为什么这么做：

- 没有 example 的接入很难真正落地
- 配置模板能直接告诉使用者 RLinf 侧需要暴露哪些最小字段

---

## 3.5 新增 DreamZero 专属训练入口和启动脚本

文件：

- [RLinf/examples/sft/train_dreamzero_sft.py](/Users/wwxq/dreamzero/RLinf/examples/sft/train_dreamzero_sft.py)
- [RLinf/examples/sft/run_dreamzero_sft.sh](/Users/wwxq/dreamzero/RLinf/examples/sft/run_dreamzero_sft.sh)
- [RLinf/examples/sft/README_dreamzero_sft.md](/Users/wwxq/dreamzero/RLinf/examples/sft/README_dreamzero_sft.md)

作用：

- 提供 DreamZero 专属训练入口
- 提供可直接复制的 bash 启动命令
- 提供一份简单的 example 文档

为什么这么做：

- 降低首次启动成本
- 避免每次都要手写 `python ... --config-path ... --config-name ...`

## 4. 为什么采用“复用 DreamZero 原配置 + RLinf 负责调度”的方案

这次没有选择“在 RLinf 里重写一整套 DreamZero 数据逻辑”，主要有四个原因。

### 4.1 DreamZero 的数据语义太重，不适合轻量重写

DreamZero 的训练逻辑并不是简单的：

- 读图
- 读 action
- 拼 batch

它还包含：

- 多视角视频组织
- 语言模板拼接
- tokenizer 处理
- 特定 embodiment 到 projector 的映射
- 相对 action / 绝对 action 逻辑

这些都已经沉淀在 DreamZero 原有 config 和 collator 里。

如果在 RLinf 里重写一遍：

- 容易出错
- 后续 DreamZero 一旦改 config，RLinf 侧也要跟着维护

### 4.2 RLinf 更适合做调度层，而不是 DreamZero 训练语义的拥有者

RLinf 的优势在于：

- 分布式 worker
- FSDP/Megatron 训练管理
- runner / logging / checkpoint 管理

DreamZero 的优势在于：

- 模型定义
- 数据 schema
- 特定训练前处理

让各自只负责自己擅长的那一层，长期维护成本最低。

### 4.3 这样更容易与 DreamZero 原始 checkpoint 兼容

如果训练路径尽量贴近 DreamZero 原实现：

- 原有 checkpoint 更容易加载
- 训练产物也更容易回到 DreamZero 的推理链路中

### 4.4 这是最适合先打通最小链路的方式

现在最重要的不是“一次性做成最终版”，而是：

- 先让 DreamZero 能在 RLinf 中稳定启动训练
- 然后再逐步补：
  - 更完整的 eval
  - 更通用的数据根路径映射
  - 更细的 resume 语义
  - 更正式的文档接入

## 5. 目前已经具备的能力

接入完成后，现在已经具备：

- RLinf 可以识别 DreamZero SFT 并启动训练
- 可以直接复用 DreamZero 原始 Hydra 训练配置
- 可以构造 DreamZero 原生 batch
- 可以让 FSDP 直接包裹底层 `VLA`
- 可以在 RLinf checkpoint 中额外保存 DreamZero 所需元信息
- DreamZero 的 action model wrapper 可以读取 RLinf 风格 `full_weights.pt`

## 6. 目前还没有完全解决的部分

这次接入重点是“最小可训练链路”，所以还有一些地方没有完全做完。

### 6.1 DreamZero 专属 eval 还没补全

当前 `FSDPVlaSftWorker.get_eval_model_output()` 仍然没有实现 DreamZero 的评估逻辑。

### 6.2 多种 DreamZero 数据配置的自动发现仍然比较保守

现在默认逻辑是：

- 优先使用 `actor.model.data_root_key`
- 否则尝试自动找到唯一一个 `*_data_root`

如果配置过于复杂，仍然建议显式指定 `data_root_key`。

### 6.3 真实训练启动还需要你本地环境验证

这次我已经做了语法检查，但还没有用真实数据和真实 checkpoint 跑完整 smoke test。

因此后续仍建议你做一次最小验证：

1. 能否成功加载 DreamZero 原始 train config
2. dataloader 是否能取到第一批数据
3. 第一轮 forward / backward 是否能通过
4. checkpoint 保存后，DreamZero wrapper 是否能重新加载

## 7. 总结

这次接入的核心思路可以概括为一句话：

`不在 RLinf 中重写 DreamZero，而是在 RLinf 中托管 DreamZero 的训练执行。`

具体来说：

- `RLinf` 提供分布式训练框架、worker、runner、FSDP 和 checkpoint 容器
- `DreamZero` 继续提供模型、数据配置、collator 和训练语义
- 我新增了一层很薄的 builder / adapter，把两者接起来

这样做的好处是：

- 改动范围可控
- 与原始 DreamZero 行为更一致
- 后续更容易继续扩展

如果后面继续推进，我建议下一步优先做这三件事：

1. 用真实 DreamZero checkpoint + DROID 数据做一次 smoke test
2. 补一个 DreamZero 专属 eval 流程
3. 再决定是否把更多 DreamZero 配置参数正式纳入 RLinf 的统一配置体系
