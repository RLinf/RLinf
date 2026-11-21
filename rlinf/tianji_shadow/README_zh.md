# Tianji Stage2 后台 Monitor 1113

适用于 tianji2 现有的 **DAgger3-All、DAgger3-Large、DAgger5-r5a-4000、DAgger5-r5b-4000、硬盒子 SFT + 上轮接管（Ablation-PrevHIL-4000）** 桌面入口。每个入口继续载入原来的 Policy，并在同一会话启动一次 Monitor。模型编码器驻留在 GPU1，微小的 Hover 输出头沿用上游 CPU 评估路径。

## 使用

1. 双击原来的 `Tianji-Stage2-30Hz-DAgger3-All_joint.desktop`、`Tianji-Stage2-30Hz-DAgger3-Large_joint.desktop`、`Tianji-Stage2-30Hz-DAgger5-r5a-4000_joint.desktop`、`Tianji-Stage2-30Hz-DAgger5-r5b-4000_joint.desktop` 或 `Tianji-Stage2-30Hz-Ablation-PrevHIL-4000_joint.desktop`。按原流程准备设备，打开 [8879 面板](http://127.0.0.1:8879/)。
2. 等待面板显示「模型状态：已加载（GPU1）」。1113 会话默认关闭 Monitor。r5b 自 1116v1 起默认开启。等待开始时，点 **Y 开启** 或按 Y 让下一条使用 Monitor。点 **N 关闭** 或按 N 让下一条停止使用。选择在当前会话内保留，Y/N 不重载模型。
3. 开始后照常按阶段按钮、数字键或 N 标注 subtask。**运行中和暂停时，N 用于下一阶段。** H/P、成功、失败、丢弃和回位沿用 Stage2。Monitor 推进、H 候选和 S4 达标作为后台预测写入。由操作员决定何时接管、推进和结束。
4. 面板显示加载失败时，可以继续人工采集。排查时查看当前 run 的 `monitor_shadow_runtime.log`。

## 模型与阈值

- 训练发布目录 `/data1/rqh/data/outputs0913/tianji_tactile_sft+hil_v8s_step10000_hover_1113`（rqh199）。
- 1113 Monitor 使用 `runs/joint/epoch_04.pt` 的 Joint 编码和基础进度头，加 `runs/hover_C/epoch_30.pt` 的进度修正和风险头。部署的纯推理权重、统计和身份在 `/home/xys/tianji_monitor_assets/shadow1113`。
- 进度阈值按 S1–S4 为 **1.00 / 2.01 / 3.02 / 3.96**，连续两项有效分数严格超过才产生完成候选。H 阈值 **0.9999223947525024**，每个连续风险段记录一次。超时阈值按 S1–S4 为 **50 / 90 / 100 / 15 秒**，计 Policy 运行时间。暂停和接管冻结计时，交回 Policy 重新计时。
- 实时输入复用当前 Policy 的 2048 维 RLT 和同一次读取的 16 维关节状态。保持最近五个连续因果观测，历史缺少用 mask。每条唯一、有效的 Policy 输出都用于判断，采集、读取、送达和判断时间保留用于分析。相邻观测间隔超过 1.2 秒重积累。触觉继续匹配采集时间不晚于观测且延迟在 46.509 ms 以内的帧。缺少有效采集时间时，使用 Policy 读取时间组织历史，本次触觉按缺失处理。
- 上游阈值在 v8s step10000 的特征上标定。报表按各入口实际 Policy 权重归类，真实目标模型上的判断性能取决于实机数据。

## 记录和重算

每条启用的轨迹在 `/home/xys/rlt_test_rtj/runs/<run>/monitor_shadow/episode_<id>_<monotonic_ns>/`。原 Stage2 数据继续写入原目录。

| 文件 | 内容 |
| --- | --- |
| `identity.json` | Monitor 源权重、校验值、规则和当次 Policy 路径 |
| `human.jsonl` | HIL 确认的阶段、回退重入、H/P、暂停恢复、结果和故障 |
| `observations.jsonl`、`obs_*.npz` | 当前 RLT、关节状态、两条五点窗口、触觉像素与掩码、时间、分数、拒绝原因 |
| `predictions.jsonl` | 与人工阶段对照的预测、独立 S1–S4 时间线、`counterfactual`、模型 H 及超时候选 |
| `comparison.json` | 阶段预测减人工锚点的有符号时间差（绝对值不超过 2 秒为吻合）。模型 H 在同次阶段进入内与人工 H 前 2 秒一对一配对，另记重复、未匹配和观测不足。 |

在 tianji2 用同一 GPU 重算进度、风险和模型预测事件。标准输出为 JSONL，标准错误输出为一致性摘要。原始超时及人工事件保存在各自日志。

```bash
export PYTHONPATH=/home/xys/rlt_test_rtj/RLinf
RUN=/home/xys/rlt_test_rtj/runs/要复查的运行目录
CUDA_VISIBLE_DEVICES=1 /home/xys/rlt_test/.venv-gpu/bin/python -B \
  -m rlinf.tianji_shadow replay "$RUN/monitor_shadow" \
  --device cuda:0 > /tmp/monitor_1113_replay.jsonl
```

## 接入其他入口和恢复

共享库位于 `/home/xys/rlt_test_rtj/RLinf/rlinf/tianji_shadow`。采用相同 Stage2 joint 环境、相同顺序的四个箱子阶段且能输出同形状 RLT/关节状态的入口，可在其 wrapper 转交 `tianji_stage2_desktop.sh` 前加 `export TIANJI_STAGE2_SHADOW_VARIANT="入口名称"`。用真实运行数据重新验证该 Policy 的判断表现。

当前会话只需在等待开始时按 N 停止判断和记录。若要恢复到集成前的程序版本，正常退出该会话后，按部署备份目录中的 `restore.sh` 恢复八个接入文件。恢复脚本核对当前文件哈希，避免覆盖之后的修改。


## 2026-11-15 验收

- 上游对齐通过。GPU1 验证 320 个锁定样本，覆盖四个采样相位、原触觉及双侧屏蔽，重现 6 次完成候选和 8 次模型 H。进度、风险概率最大绝对误差分别为 4.77×10⁻⁷ 和 8.05×10⁻⁷。
- 三个桌面入口的 wrapper 及权重路径通过无动作验证。模拟 8879 流程检查了预加载、默认关闭、Y/N 保持 PID、关闭后无新增数据、运行中及暂停时的阶段切换、H/P、回退/重入和归档。故障模拟验证了缺少 Monitor 权重时人工 HIL 可继续。基于同一 GPU 的模拟轨迹重算分数与录制一致。
- 规则单测覆盖了严格阈值、间隔重置、暂停期间冻结超时、独立 S1–S4 时间线、人工时间对照、触觉因果性和设备换绑。
- 实机仍待操作员从三个入口各采一条。每条需要核对 Policy 延迟和 GPU1 训练共载、真实观测的可用比例、触觉覆盖、对齐时间和记录文件完整性。

本次备份目录为 `/home/xys/tianji_monitor_assets/backups/shadow1113_20261115_215334`，其中的 `deployment.json` 保存部署与原始文件校验。无动作验证记录位于 `/home/xys/tianji_monitor_assets/shadow1113_stage` 的 `final_gpu_verify.json`、`final_rules.log` 和 `final_simulation.log`。


## 2026-11-16 r5b 接入

原 `Tianji-Stage2-30Hz-DAgger5-r5b-4000_joint.desktop` 从下一次启动起复用同一套 Monitor。Policy 使用 `/home/xys/rlt_models/box_r5b_step4000`，运行日志标识为 `DAgger5-r5b-4000`。启动脚本的语法、模型路径及参数传递已做无动作验证。r5a 和 r5b 已用正式部署包通过 GPU1 无动作验证，覆盖单次预加载、默认关闭、Y/N 切换、H/P 及归档。实机采集待验证。

若只需回退 r5b 接入，正常退出 r5b 会话后，将 `/home/xys/tianji_monitor_assets/backups/shadow1113_r5b_20261116_095729/tianji_stage2_r5b_4000.sh` 恢复到 `/home/xys/rlt_test_rtj/launcher/tianji_stage2_r5b_4000.sh`。

## 2026-11-16 硬盒子 SFT + 上轮接管接入

原 `Tianji-Stage2-30Hz-Ablation-PrevHIL-4000_joint.desktop` 从下一次启动起复用同一套 Monitor。Policy 使用 `/home/xys/rlt_models/box_r4q_step4000`，后台记录标识为 `Ablation-PrevHIL-4000`。启动脚本、模型和提示词路径通过无动作验证。GPU1 模拟流程验证了单次预加载、默认关闭、Y/N、H/P 和记录归档，同卡重算分数一致，实机采集待验证。

若只需回退该入口的接入，正常退出会话后，将 `/home/xys/tianji_monitor_assets/backups/shadow1113_r4q_20261116_100740/tianji_stage2_r4q.sh` 恢复到 `/home/xys/rlt_test_rtj/launcher/tianji_stage2_r4q.sh`。

### r5b 的 episode 复核文件

DAgger5 版本二 4000 步从下一次启动起，在每次实际开始时创建
`<run>/episode_reviews/episode_<id>_<start_ns>/review.jsonl`。
人工切段（含回退和重进）、H、P 和结束结果均记录。开始行的
`monitor_enabled` 标明本条是否开启后台判断。开启时，同一文件还记录
Monitor 的独立阶段完成、按人工当前阶段计算的完成候选、模型 H 和超时 H。
`source` 区分人工和 Monitor，`timeline` 标明 Monitor 使用的时间线。
`ns` 为同一单调时钟上的事件时间，合并回放时按它排序。模型行的 `input`
相对于当前 run，可定位到用于该次判断的已保存输入。旧轨迹保留原始日志。

Stage2 joint 的共用启动器默认允许 0、1 或 2 只 Xense 在线，适用于新训练、续训和各模型入口。
触觉优化版采用同样的启动规则。采集器保持 Xense 开启，在线通道继续采集，
缺失通道的模型有效掩码为 false。机器人反馈、相机和控制状态仍需通过启动检查。
早期非 joint 入口也启用采集器的 Xense 功能，其 Policy 仍为视觉模型。
8879 面板底部每约 2 秒显示一次 Xense 在线数量。按设备连接状态计数，
因此已连接但尚未产生有效触觉的设备也计入。采集器状态缺失时显示「状态暂不可用」。


## r5b 使用 1116v1

原 DAgger5 版本二 4000 步桌面图标启动后，「后台 Monitor」旁显示 1116v1，默认 Y 开启。等候时按 N 可关闭，轨迹运行及暂停时 N 继续用于推进人工阶段。Policy 为 box_r5b_step4000。其他入口使用 1113。

- 单点阈值 S1–S4 为 1.02 / 2.02 / 3.07 / 3.96，双点阈值为 1.00 / 2.01 / 3.04 / 3.94，严格超阈。
- 每条轨迹同时记录 single 和 two。每套规则保留按人工阶段对照和独立时间线。模型 H 阈值为 0.9988200068473816，单个连续风险段产生一次候选，超时 H 另计。
- 1116v1 使用 /home/xys/tianji_monitor_assets/shadow1116。模型来自 rqh199 上的 tianji_tactile_sft+hil_1116 的 Joint E4 与 Hover C E30。

既可离线复查 1116v1 原始轨迹，也可从本日 r5b 的已保存 Policy 输入重建两条规则。为每次复算使用独立输出目录。原 1113 仍通过 replay 子命令复算。

```bash
export PYTHONPATH=/home/xys/rlt_test_rtj/RLinf
CUDA_VISIBLE_DEVICES=1 /home/xys/rlt_test/.venv-gpu/bin/python -B   -m rlinf.tianji_shadow replay1116 --output /tmp/shadow1116-replay   /home/xys/rlt_test_rtj/runs/<run>/monitor_shadow/<episode>
```

## 在 8879 面板修改阶段阈值

r5b 的 1116v1 面板在 Y/N 下方显示「阶段完成阈值」。等待开始时可分别编辑单点和双点的 S1–S4，范围 0–4，每组递增。保存后下一条生效，也会在下次启动时载入。轨迹运行或暂停期间，面板只显示当前设置。按钮「恢复 1116v1 初始值」可恢复此前部署的两组阈值。

每条 Monitor 轨迹的 identity.json 记录实际使用的 rule_profiles，离线复算自动使用这份快照。若需试算另一组阈值，可在 replay1116 命令增加 `--thresholds 路径.json`，文件格式示例见 `/home/xys/tianji_monitor_assets/shadow1116_operator_thresholds.json`。
