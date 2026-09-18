# SO-101 Real-World MVP 版本历史

## 时间线与版本关系

### Phase 1: 基础设施 (PR1560)
- **e1d1bac1** - PR1560 基础提交
  - 实现了 SO-101 基础支持
  - UVC 相机后端
  - 基础遥操逻辑
  - **状态**: ✅ 已合并到 main，作为 SO-101 的基础

### Phase 2: 独立 DAgger 脚本 (PR1572)
- **896d52c2** - "feat(so101): independent DAgger toolchain"
  - 创建了独立的 DAgger 脚本 (`toolkits/realworld_check/so101_dagger.py`)
  - **原因**: 发现官方 HG-DAgger 硬编码了 PICO，无法兼容 SO-101
  - **问题**: 教授反馈"不要独立实现，会让项目散掉"
  - **状态**: ❌ 被废弃

### Phase 3: 尝试接入官方框架 (被叫停)
- **stash@{0}** - "WIP: PR1572 partial fixes - backup before restart"
  - ✅ 修复了 `smooth_intervene.py` 的 PICO 硬编码
  - ✅ 实现了完整的 SO-101 HG-DAgger 接入
  - ❌ 但改动了 PR1560 的大部分代码（违反初衷）
  - **状态**: 🔒 已 stash 保存，需提取必要部分

### Phase 4: 从 PR1560 重新开始 (当前)
- **feat/so101-realworld-mvp** 分支
  - 基于 PR1560 重新实现
  - 目标: 只添加 HG-DAgger 接入所需的最小改动
  - **当前状态**: 🚧 进行中

### Phase 5: Worktree 中的 Milestone 3 (孤立)
- **ad64d217** (worktree-milestone3-hg-dagger)
  - 另一个 Claude 会话完成的实现
  - ⚠️ 没有修复 `smooth_intervene.py` 的核心问题
  - **状态**: ❌ 无法运行，未合并

---

## Milestone 标记

### Milestone 1: 基础数据收集 ✅
- **Commit**: `1ae42a8e` (tag: milestone-1-ready)
- **分支**: feat/so101-realworld-mvp
- **功能**: 
  - SO-101 环境初始化
  - 数据收集配置
  - 验证 schema
- **状态**: ✅ 开发完成，等待真机测试

### Milestone 2: gRPC / Local Inference（推理功能）✅
- **Commit**: `baa76ff8` → `5b0d75e1` (feat/so101-realworld-mvp-clean)
- **功能**:
  - GRPCPolicyAdapter - 远程推理适配器
  - gRPC server 实现 (lerobot_grpc_policy_server.py)
  - 本地推理 / gRPC 云端推理两种模式
  - GPU 优化支持（torch.compile, CUDA graphs）
  - YAML 配置重构（对齐 Franka 结构）
  - 推理相关文档 (GRPC_IMPLEMENTATION.md)
- **状态**: ✅ 开发完成，等待真机测试

### Milestone 3: HG-DAgger 🚧
- **Commit**: 进行中
- **目标**:
  - ✅ `SO101Leader.hold()` 方法
  - ⚠️ 修复 `smooth_intervene.py` PICO 限制（从 stash 提取）
  - ✅ HG-DAgger 配置文件
  - ⚠️ 不改动 PR1560 的已有代码
- **状态**: 🚧 进行中

---

## 关键技术问题

### 核心障碍: PICO 硬编码
**位置**: `rlinf/utils/env_helpers/smooth_intervene.py`

**原始代码**:
```python
if not named or any(name != "pico" for name in named):
    raise ValueError("smooth_intervene requires every env.train.teleop entry to be pico")
```

**stash@{0} 的修复**:
```python
if mode == "activity" and any(name != "pico" for name in named):
    raise ValueError(
        "activity-triggered smooth_intervene requires every "
        f"env.train.teleop entry to be pico (got {named!r}); use "
        "teleop_intervention.mode: explicit for stateful devices"
    )
```

**关键点**:
- 只在 `mode == "activity"` 时限制 PICO
- 允许 `mode: explicit` 用于 SO-101 等有状态设备

---

## 下一步行动

1. **提取 stash@{0} 的必要改动**
   - smooth_intervene.py 的修复
   - SO101Leader.hold() 方法
   - 配置文件（如果不影响 PR1560）

2. **标记 Milestone 1 & 2 的 commits**
   - 用 git tag 标记关键节点
   - 便于真机测试时快速切换

3. **完成 Milestone 3**
   - 只做最小必要改动
   - 不重构 PR1560 的代码
