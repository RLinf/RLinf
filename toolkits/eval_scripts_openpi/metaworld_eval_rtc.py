# toolkits/eval_scripts_openpi/metaworld_eval_rtc.py
#
# 这份脚本是“MetaWorld + pi_0 + RTC 风格异步执行”的独立评测入口。
# 和官方 metaworld_eval.py 相比，它不再只在 action_plan 为空时同步补货，
# 而是让前台按固定 control_hz 消费动作，同时让后台线程提前重规划下一段 chunk。
#
# 这一版按 RTC 论文 Algorithm 1 组织：
# 1. InitializeSharedState: 初始化 t、A_cur、o_cur、delay queue Q；
# 2. GetAction: 前台控制器每隔 Δt 更新最新观测，并返回 A_cur[t-1]；
# 3. InferenceLoop: 后台线程等待 t >= s_min 后启动 GuidedInference；
# 4. GuidedInference: 复用 pi0 denoising loop，并用旧 chunk 剩余动作做 overlap guidance；
# 5. Swap: 新 chunk 一可用就替换 A_cur，并执行 t = t - s 自动跳过过期前缀。

# Copyright 2025 The RLinf Authors.
# Modified for inference-only RTC-style asynchronous control on MetaWorld.

import argparse
import collections
import copy
import json
import math
import os
import pathlib
import threading
import time

# 尽量在导入 MuJoCo 相关库之前就固定渲染后端。
# 在无显示器的服务器上，这通常比默认后端更稳定。
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import imageio
import metaworld
import numpy as np

from toolkits.eval_scripts_openpi import setup_logger, setup_policy
from toolkits.eval_scripts_openpi.rtc_guided_sampling import (
    RTCConfig,
    guided_sample_actions,
    postprocess_actions_from_policy,
    prepare_observation_from_policy,
)

metaworld.register_mw_envs()

PROMPT_JSON_PATH = "rlinf/envs/metaworld/metaworld_config.json"


def load_prompt_config(json_path):
    """
    读取 MetaWorld 任务描述和难度分组配置。

    返回:
    - task_description_dict: 任务名 -> prompt 字符串
    - difficulty_to_tasks: 难度名 -> 任务列表
    """
    with open(json_path, "r") as f:
        config_data = json.load(f)
    task_description_dict = config_data.get("TASK_DESCRIPTIONS", {})
    difficulty_to_tasks = config_data.get("DIFFICULTY_TO_TASKS", {})
    return task_description_dict, difficulty_to_tasks


TASK_DESCRIPTION_DICT, DIFFICULTY_TO_TASKS = load_prompt_config(PROMPT_JSON_PATH)
ENV_LIST = list(TASK_DESCRIPTION_DICT.keys())

def deepcopy_batch(batch):
    """
    对 policy 输入 batch 做安全拷贝。

    numpy 数组单独 copy 底层数据，避免前台/后台线程共享同一块内存。
    """
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = np.array(v, copy=True)
        else:
            out[k] = copy.deepcopy(v)
    return out


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def format_metric(value, digits=3):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class AsyncRTCExecutor:
    """
    和图中 Algorithm 1 对齐的 Real-Time Chunking 执行器。

    - M：self._condition 内部的锁，保护共享变量；
    - C：self._condition，用来让后台 InferenceLoop 等待前台 GetAction 通知；
    - t：self.t，表示从当前 A_cur 被启用以来，前台已经索引/执行到第几步；
    - A_cur：self.A_cur，当前正在执行的 chunk，shape 逻辑上是 [H, A_env]；
    - o_cur：self.o_cur，最近一个控制周期看到的观测 batch；
    - Q：self.delay_queue，历史推理 delay buffer，里面每个元素单位是“控制步”。

    张量/数组维度约定：
    - H：prediction horizon，也就是 chunk 长度；MetaWorld pi0 通常 H=5。
    - A_env：环境动作维度；MetaWorld 通常 A_env=4。
    - 单步 action：np.ndarray [A_env]。
    - A_cur / A_new：list[np.ndarray]，整体等价于 [H, A_env]。
    """

    def __init__(
        self,
        policy,
        action_chunk,
        control_dt,
        min_exec_horizon=2,
        delay_buffer_size=10,
        num_denoise_steps=5,
        guidance_clip=5.0,
        initial_delay_steps=1,
        inject_delay_ms=0.0,
        logger=None,
        runtime_log=False,
    ):
        self.policy = policy
        # action_chunk 在这个脚本中对应 Algorithm 1 的 prediction horizon H。
        # 执行前是用户传入的 int；执行后仍是 int，只是强制转成 Python 标量。
        self.action_chunk = int(action_chunk)
        # control_dt 是固定控制周期 Δt，单位秒；例如 control_hz=10 时 control_dt=0.1。
        self.control_dt = float(control_dt)
        # s_min：后台至少等前台执行这么多步后，才启动下一次 GuidedInference。
        self.min_exec_horizon = int(min_exec_horizon)
        # Q 的最大长度 b。Q 中存的是历史 delay steps，不是毫秒。
        self.delay_buffer_size = int(delay_buffer_size)
        # d_init：初始 delay 估计，Algorithm 1 中 Q = Queue([d_init], maxlen=b)。
        self.initial_delay_steps = max(0, int(initial_delay_steps))
        self.inject_delay_ms = float(inject_delay_ms)

        self.logger = logger
        self.runtime_log = bool(runtime_log)

        # RTCConfig 会传给 guided_sample_actions，对应 Algorithm 1 的 n、s_min、β、b。
        self.rtc_cfg = RTCConfig(
            num_steps=int(num_denoise_steps),
            min_exec_horizon=self.min_exec_horizon,
            guidance_clip=float(guidance_clip),
            delay_buffer_size=self.delay_buffer_size,
        )

        # M + C：Condition 同时包含 mutex 和条件变量。
        # 所有共享状态 t / A_cur / o_cur / Q 都必须在这把锁下读写。
        self._condition = threading.Condition()
        self._stop = False
        self._initialized = False

        # Algorithm 1: t = 0, A_cur = A_init, o_cur = null。
        # 初始化前 A_cur 为空；第一次 step(obs) 会同步计算 A_init。
        self.t = 0
        self.A_cur = []
        self.o_cur = None
        self.delay_queue = collections.deque(
            [self.initial_delay_steps], maxlen=self.delay_buffer_size
        )

        # 后台 InferenceLoop 是一个常驻线程，里面循环等待 condition C，并在满足 t >= s_min 时启动 GuidedInference。
        self._thread = None
        self._inference_running = False

        # 下面这些字段主要用于日志和后处理分析。
        self.last_infer_ms = None
        self.last_predicted_delay_steps = None
        self.last_observed_delay_steps = None
        self.last_time_delay_steps = None
        self.last_s = None
        self.deadline_miss_count = 0
        self.bootstrap_infer_ms = None
        self.bootstrap_infer_ms_sum = 0.0
        self.bootstrap_infer_count = 0
        self.guided_infer_ms_sum = 0.0
        self.guided_infer_count = 0
        self.all_infer_ms_sum = 0.0
        self.all_infer_count = 0

    def close(self):
        """停止后台 InferenceLoop，避免 episode 结束后还有线程继续访问同一个 policy。"""
        with self._condition:
            self._stop = True
            self._condition.notify_all()

        if self._thread is not None:
            # 等后台当前这次 GuidedInference 正常结束；这样下一个 episode 不会和旧线程抢模型。
            self._thread.join(timeout=60.0)
            if self._thread.is_alive():
                self._log("[RTC] warning: background inference thread did not stop within 60s.")

    def _log(self, msg):
        if self.runtime_log and self.logger is not None:
            self.logger.info(msg)

    def get_timing_summary(self):
        with self._condition:
            return {
                "bootstrap_infer_ms": self.bootstrap_infer_ms,
                "bootstrap_infer_ms_sum": self.bootstrap_infer_ms_sum,
                "bootstrap_infer_count": self.bootstrap_infer_count,
                "avg_bootstrap_infer_ms": safe_div(
                    self.bootstrap_infer_ms_sum, self.bootstrap_infer_count
                ),
                "guided_infer_ms_sum": self.guided_infer_ms_sum,
                "guided_infer_count": self.guided_infer_count,
                "avg_guided_infer_ms": safe_div(
                    self.guided_infer_ms_sum, self.guided_infer_count
                ),
                "all_infer_ms_sum": self.all_infer_ms_sum,
                "all_infer_count": self.all_infer_count,
                "avg_infer_ms": safe_div(
                    self.all_infer_ms_sum, self.all_infer_count
                ),
                "deadline_miss_count": self.deadline_miss_count,
            }

    @staticmethod
    def _to_action_list(actions):
        """
        把 policy 输出统一整理成 list[np.ndarray]。

        执行前：
        - actions 可能是 np.ndarray [H, A_env]，也可能已经是 list，每个元素 [A_env]。

        执行后：
        - 返回 list[np.ndarray]，长度为 H；
        - 每个元素是 float32 np.ndarray [A_env]。
        """
        return [np.asarray(a, dtype=np.float32) for a in actions]

    def _sync_initial_chunk(self, obs):
        """
        计算 Algorithm 1 里的 A_init。

        输入 obs 的典型格式：
        {
            "observation/image": np.ndarray [H_img, W_img, 3],
            "observation/state": np.ndarray [4],
            "prompt": str,
        }

        输出：
        - actions: list[np.ndarray]，整体等价于 [H, A_env]，MetaWorld 下 A_env=4。
        """
        infer_t0 = time.perf_counter()
        out = self.policy.infer(deepcopy_batch(obs))
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        actions = self._to_action_list(out["actions"][: self.action_chunk])
        if len(actions) == 0:
            raise RuntimeError("policy.infer returned empty A_init for RTC.")
        self.bootstrap_infer_ms = infer_ms
        self.bootstrap_infer_ms_sum += infer_ms
        self.bootstrap_infer_count += 1
        self.all_infer_ms_sum += infer_ms
        self.all_infer_count += 1
        return actions

    def _prepare_prev_remaining_for_guidance(self, obs_snapshot, A_prev_env):
        """
        把 Algorithm 1 里的 A_prev 从环境动作空间转成 pi0 模型动作空间。

        为什么需要这一步：
        - A_cur / A_prev 来自 policy output_transforms 之后，MetaWorld 下是环境动作 [H-s, 4]；
        - guided_sample_actions 内部的 x_t / target 在 pi0 模型域里，形状是 [1, H, A_model]；
        - 因此直接拿 [H-s, 4] 去约束 [1, H, A_model] 既可能维度不匹配，也可能归一化尺度不一致。

        输入：
        - A_prev_env: np.ndarray [L, A_env]，L=H-s，MetaWorld 下 A_env=4。

        输出：
        - A_prev_model: np.ndarray [L, A_model]，已经经过 input transforms 的动作目标。
        """
        A_prev_env = np.asarray(A_prev_env, dtype=np.float32)
        if len(A_prev_env) == 0:
            return A_prev_env

        # 构造一个带 actions 的临时 batch，让 OpenPI 原有 input transform 负责：
        # 1. action normalization；
        # 2. action padding；
        # 3. 其他和模型配置相关的动作域转换。
        obs_with_actions = deepcopy_batch(obs_snapshot)
        obs_with_actions["actions"] = np.array(A_prev_env, copy=True)
        # 有些数据配置习惯使用 "action" 作为原始字段名；两个 key 都放进去更稳妥。
        obs_with_actions["action"] = np.array(A_prev_env, copy=True)

        try:
            transformed = self.policy._input_transform(obs_with_actions)
            if "actions" in transformed:
                A_prev_model = np.asarray(transformed["actions"], dtype=np.float32)
            elif "action" in transformed:
                A_prev_model = np.asarray(transformed["action"], dtype=np.float32)
            else:
                raise KeyError("input transform output does not contain actions/action")

            # 通常 input_transform 后 actions 是 [L, A_model]；
            # 如果某个 transform 额外带了 batch 维 [1, L, A_model]，这里去掉 batch 维。
            if A_prev_model.ndim == 3 and A_prev_model.shape[0] == 1:
                A_prev_model = A_prev_model[0]

            return A_prev_model
        except Exception as e:
            # 兜底路径：如果动作 transform 失败，至少把 [L, A_env] 右侧补零到 [L, A_model]。
            # 这能避免维度崩溃，但归一化尺度不如上面的 transform 路径严格。
            model_action_dim = int(self.policy._model.config.action_dim)
            A_prev_model = np.zeros(
                (A_prev_env.shape[0], model_action_dim), dtype=np.float32
            )
            copy_dim = min(A_prev_env.shape[1], model_action_dim)
            A_prev_model[:, :copy_dim] = A_prev_env[:, :copy_dim]
            self._log(
                "[RTC] warning: failed to transform A_prev into model action space; "
                f"falling back to zero-padding. error={repr(e)}"
            )
            return A_prev_model

    def _initialize_shared_state(self, obs):
        """
        对应 Algorithm 1 的 InitializeSharedState。

        执行前：
        - 还没有 A_cur，前台没有动作可执行。

        执行后：
        - self.t = 0；
        - self.A_cur = A_init，整体等价于 [H, A_env]；
        - self.o_cur = null/None，随后第一次 GetAction 会填入最新观测；
        - 后台 InferenceLoop 线程启动并等待 condition C。
        """
        A_init = self._sync_initial_chunk(obs)

        with self._condition:
            self.t = 0
            self.A_cur = A_init
            self.o_cur = None
            self.delay_queue = collections.deque(
                [self.initial_delay_steps], maxlen=self.delay_buffer_size
            )
            self._initialized = True

            self._thread = threading.Thread(
                target=self._inference_loop,
                name="rtc-inference-loop",
                daemon=True,
            )
            self._thread.start()

    def _guided_inference(self, obs_snapshot, A_prev, delay_steps, s):
        """
        对应 Algorithm 1 的 GuidedInference(pi, o, A_prev, d, s)。

        输入：
        - obs_snapshot: 后台启动时拷贝的最新观测 o；
        - A_prev: 旧 chunk 尚未被执行的尾部，shape 等价于 [H-s, A_env]；
        - delay_steps: d = max(Q)，保守估计的新 chunk 过期前缀长度；
        - s: 启动推理时已经执行的步数，用于日志；A_prev 已经按 s 裁好。

        内部流程：
        1. prepare_observation_from_policy 复用 OpenPI Policy input transforms；
        2. guided_sample_actions 在 pi0 denoising loop 里加入 RTC guidance；
        3. postprocess_actions_from_policy 再复用 output transforms，把模型域动作还原成环境动作。

        输出：
        - A_new: list[np.ndarray]，整体等价于 [H, A_env]。
        """
        if self.inject_delay_ms > 0:
            # 人工延迟只用于仿真实验，模拟“推理结果晚于控制周期”的情况。
            time.sleep(self.inject_delay_ms / 1000.0)

        # inputs 是 transform 后的模型输入字典；observation 是 OpenPI 的 Observation。
        # 典型维度变化：
        # state: [4] -> [1, 4]；
        # image: [H_img, W_img, 3] -> [1, H_img, W_img, 3] 或 transform 后的等价布局。
        inputs, observation = prepare_observation_from_policy(self.policy, obs_snapshot)

        # A_prev 执行前是环境动作维 [H-s, A_env]；
        # 这里先转成模型动作域 [H-s, A_model]，再交给 guided_sample_actions。
        A_prev_model = self._prepare_prev_remaining_for_guidance(
            obs_snapshot=obs_snapshot,
            A_prev_env=A_prev,
        )

        # guided_sample_actions 内部会构造 target/mask，并在模型动作维 [1, H, A_model] 上做 guidance。
        actions_torch = guided_sample_actions(
            policy=self.policy,
            observation=observation,
            prev_remaining=A_prev_model,
            delay_steps=int(delay_steps),
            rtc_cfg=self.rtc_cfg,
        )

        # actions_torch: [1, H, A_model] -> outputs["actions"]: [H, A_env]。
        outputs = postprocess_actions_from_policy(self.policy, inputs, actions_torch)
        A_new = self._to_action_list(outputs["actions"][: self.action_chunk])

        if len(A_new) == 0:
            raise RuntimeError("GuidedInference returned an empty A_new.")

        self._log(
            f"[RTC] GuidedInference finished with s={s}, "
            f"d_hat={delay_steps}, A_prev_len={len(A_prev)}, A_new_len={len(A_new)}"
        )
        return A_new

    def _inference_loop(self):
        """
        对应 Algorithm 1 的 InferenceLoop，运行在后台线程中。

        关键点：
        - 后台线程不会每个控制周期都推理；
        - 它会一直等待 condition C，直到前台 t >= s_min；
        - 启动推理时取快照：s、A_prev、o、d；
        - 推理期间释放 mutex M，让前台 GetAction 继续实时执行；
        - A_new 可用后重新拿锁，立刻把 A_cur 换成 A_new，并执行 t = t - s。
        """
        while True:
            with self._condition:
                # wait on C until t >= s_min。
                # 执行前：后台线程可能刚启动，o_cur=None 或 t 还太小；
                # 执行后：保证有最新观测，并且至少执行了 s_min 步旧 chunk。
                while (
                    not self._stop
                    and (self.o_cur is None or self.t < self.min_exec_horizon)
                ):
                    self._condition.wait()

                if self._stop:
                    return

                # s = t：这次推理启动时，旧 chunk 已经执行了 s 步。
                s = int(self.t)

                # A_cur 等价于 [H, A_env]；A_prev = A_cur[s:H] 等价于 [H-s, A_env]。
                # 这一步正对应论文伪代码里的 “Remove the s actions that have already been executed”。
                A_prev = np.asarray(self.A_cur[s:], dtype=np.float32)

                # o = o_cur：取最新观测快照，释放锁后后台只读这份副本。
                obs_snapshot = deepcopy_batch(self.o_cur)

                # d = max(Q)：用历史 delay buffer 做保守估计。
                delay_steps = int(max(self.delay_queue)) if self.delay_queue else 0

                self._inference_running = True
                self.last_s = s
                self.last_predicted_delay_steps = delay_steps

            # with M released do GuidedInference(...)。
            # 这里不能持有 self._condition 的锁，否则前台控制循环会被后台推理阻塞。
            t0 = time.perf_counter()
            try:
                A_new = self._guided_inference(
                    obs_snapshot=obs_snapshot,
                    A_prev=A_prev,
                    delay_steps=delay_steps,
                    s=s,
                )
            except Exception as e:
                self._log(f"[RTC] GuidedInference failed: {repr(e)}")
                with self._condition:
                    self._inference_running = False
                time.sleep(self.control_dt)
                continue

            infer_ms = (time.perf_counter() - t0) * 1000.0
            # 这是按墙钟时间估计的 delay steps，只用于日志；
            # Algorithm 1 真正入队的是下面 observed_delay_steps = t - s。
            time_delay_steps = int(math.ceil(infer_ms / (self.control_dt * 1000.0)))

            with self._condition:
                if self._stop:
                    return

                self.guided_infer_ms_sum += infer_ms
                self.guided_infer_count += 1
                self.all_infer_ms_sum += infer_ms
                self.all_infer_count += 1

                # 推理期间前台仍在调用 GetAction，因此 self.t 已经从 s 继续增长。
                # observed_delay_steps = self.t - s 表示“这次推理期间又执行了多少个控制步”。
                observed_delay_steps = max(0, int(self.t) - s)

                # A_cur = A_new：新 chunk 一旦可用就立即替换。
                # 执行前 A_new 是 list[np.ndarray]，整体 [H, A_env]；
                # 执行后 self.A_cur 仍是 [H, A_env] 语义，只是内容换成新规划。
                self.A_cur = A_new

                # t = t - s：把索引重置为“进入新 chunk 后已经过期/应该跳过的前缀长度”。
                # 后续 GetAction 会先 t += 1，再返回 A_cur[t - 1]，
                # 因此这一步自然实现了“跳过推理期间已经错过的新 chunk 前缀”。
                self.t = observed_delay_steps

                # enqueue t onto Q：记录本次观测到的 delay，供下一次 d=max(Q) 使用。
                self.delay_queue.append(observed_delay_steps)

                self.last_infer_ms = infer_ms
                self.last_observed_delay_steps = observed_delay_steps
                self.last_time_delay_steps = time_delay_steps
                self._inference_running = False

                self._log(
                    f"[RTC] swap A_cur: infer_ms={infer_ms:.1f}ms, "
                    f"time_delay_steps={time_delay_steps}, "
                    f"observed_delay_steps={observed_delay_steps}, "
                    f"next_t={self.t}, Q={list(self.delay_queue)}"
                )

                # swap 后如果 t 仍然已经 >= s_min，通知自己下一轮可以继续推理。
                self._condition.notify_all()

    def step(self, obs):
        """
        对应 Algorithm 1 的 GetAction(o_next)。

        前台控制器每隔 Δt 调一次这个函数。

        输入：
        - obs: 当前周期观测 o_next，格式是 policy.infer 期望的原始 batch。

        状态变化：
        - t: int -> int + 1；
        - o_cur: 上一次观测 -> 当前观测深拷贝；
        - notify C: 唤醒后台 InferenceLoop；
        - 返回 A_cur[t - 1]，也就是当前周期要执行的单步动作 [A_env]。

        返回:
        - action:
          当前周期应立刻送给环境的单步动作，MetaWorld 下通常 shape=[4]
        - stats:
          与本次 RTC 状态有关的调试统计量
        """
        if not self._initialized:
            self._initialize_shared_state(obs)

        with self._condition:
            # t = t + 1。
            # 执行前 t 表示当前 A_cur 已经索引到的位置；
            # 执行后 action_index = t - 1 正好是本周期要返回的动作下标。
            self.t += 1
            action_index = self.t - 1

            # o_cur = o_next。这里做深拷贝，避免后台线程读到外部还在变化的对象。
            self.o_cur = deepcopy_batch(obs)

            # notify C：提醒后台“新观测来了，且 t 可能已经满足 t >= s_min”。
            self._condition.notify_all()

            deadline_miss = False
            if 0 <= action_index < len(self.A_cur):
                # 正常情况：A_cur[action_index] 是单步动作 [A_env]。
                action = np.asarray(self.A_cur[action_index], dtype=np.float32)
            else:
                # 实验保护：如果推理太慢导致 t 超过 H，Algorithm 1 的索引会越界。
                # 真机上这属于 deadline miss；仿真里先保持最后一个动作，避免环境直接崩掉。
                deadline_miss = True
                self.deadline_miss_count += 1
                action = np.asarray(self.A_cur[-1], dtype=np.float32)

            remaining_actions = max(0, len(self.A_cur) - self.t)

            stats = {
                "remaining_actions": remaining_actions,
                "last_infer_ms": self.last_infer_ms,
                # 兼容旧日志字段：actual/stale 在 Algorithm 1 里都对应 t=t-s 后跳过的新 chunk 前缀。
                "last_actual_delay_steps": self.last_observed_delay_steps,
                "last_stale_skip_steps": self.last_observed_delay_steps,
                "last_predicted_delay_steps": self.last_predicted_delay_steps,
                "last_time_delay_steps": self.last_time_delay_steps,
                "last_s": self.last_s,
                "rtc_t": self.t,
                "action_index": action_index,
                "delay_queue": list(self.delay_queue),
                "deadline_miss": deadline_miss,
                "deadline_miss_count": self.deadline_miss_count,
                "has_inflight_future": self._inference_running,
            }

        return action, stats


def make_env(env_name):
    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        render_mode="rgb_array",
        camera_id=2,
        disable_env_checker=True,
    )

    # 和官方脚本保持一致。
    # 由于 MetaWorld 包装层比较深，这里继续沿用原始脚本中的多层 env 访问方式。
    try:
        env.env.env.env.env.env.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    except Exception:
        pass

    return env


def build_policy_input(observation, env_name, env):
    """
    把环境原始观测组装成 policy.infer 期望的 batch。

    执行前:
    - observation: MetaWorld 返回的扁平状态向量，shape 约为 [obs_dim]
    - env.render(): 当前帧图像，shape [H, W, 3]

    执行后:
    - batch["observation/image"]: [H, W, 3]
    - batch["observation/state"]: [4]
    - batch["prompt"]: str
    """
    image = env.render()

    # 与官方脚本保持一致，做方向翻转。
    image = image[::-1, ::-1]

    # 仅取前 4 维状态，与 pi_0 MetaWorld 训练配置保持一致。
    state = observation[:4]

    batch = {
        "observation/image": image,
        "observation/state": state,
        "prompt": TASK_DESCRIPTION_DICT[env_name],
    }
    return batch, image


def run_one_episode(env, env_name, args, logger, rtc_executor):
    frames = []
    observation, info = env.reset()

    # reset 后先喂若干个全 0 动作，让物体稳定下来。
    dummy_action = [0.0] * 4
    for _ in range(args.settle_steps):
        observation, _, _, _, _ = env.step(dummy_action)

    success = 0
    episode_stats = []
    first_success_time_s = None
    first_success_step = None
    episode_t0 = time.perf_counter()

    for step in range(args.max_steps):
        t0 = time.perf_counter()

        # build_policy_input 返回的 batch 会被送入 rtc_executor。
        batch, image = build_policy_input(observation, env_name, env)
        action, rtc_stats = rtc_executor.step(batch)

        # action 是一个单步动作向量，MetaWorld 下通常 shape=[4]。
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(image)

        step_stat = {
            "step": step,
            "success_flag": int(info.get("success", 0)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "remaining_actions": rtc_stats["remaining_actions"],
            "last_infer_ms": rtc_stats["last_infer_ms"],
            "last_actual_delay_steps": rtc_stats["last_actual_delay_steps"],
            "last_stale_skip_steps": rtc_stats["last_stale_skip_steps"],
            "last_predicted_delay_steps": rtc_stats["last_predicted_delay_steps"],
            "last_time_delay_steps": rtc_stats["last_time_delay_steps"],
            "last_s": rtc_stats["last_s"],
            "rtc_t": rtc_stats["rtc_t"],
            "action_index": rtc_stats["action_index"],
            "delay_queue": rtc_stats["delay_queue"],
            "deadline_miss": rtc_stats["deadline_miss"],
            "deadline_miss_count": rtc_stats["deadline_miss_count"],
            "has_inflight_future": rtc_stats["has_inflight_future"],
        }
        episode_stats.append(step_stat)

        if args.verbose_rtc and (rtc_stats["last_infer_ms"] is not None):
            logger.info(
                f"[{env_name}] step={step} "
                f"infer_ms={rtc_stats['last_infer_ms']:.1f} "
                f"d_hat={rtc_stats['last_predicted_delay_steps']} "
                f"observed_delay={rtc_stats['last_actual_delay_steps']} "
                f"time_delay={rtc_stats['last_time_delay_steps']} "
                f"s={rtc_stats['last_s']} "
                f"t={rtc_stats['rtc_t']} "
                f"remain={rtc_stats['remaining_actions']} "
                f"miss={rtc_stats['deadline_miss']} "
                f"inflight={rtc_stats['has_inflight_future']}"
            )

        # 固定控制周期:
        # 无论 policy 推理快慢，前台都尽量以 1 / control_hz 的节奏推进。
        elapsed = time.perf_counter() - t0
        sleep_s = max(0.0, (1.0 / args.control_hz) - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)

        if first_success_time_s is None and info.get("success", 0):
            first_success_time_s = time.perf_counter() - episode_t0
            first_success_step = step + 1

        if info.get("success", 0):
            success = 1
            if args.break_on_success:
                break

        if terminated or truncated:
            break

    episode_elapsed_s = time.perf_counter() - episode_t0
    episode_metrics = {
        "success": int(success),
        "episode_elapsed_s": episode_elapsed_s,
        "first_success_time_s": first_success_time_s,
        "first_success_step": first_success_step,
    }
    return success, frames, episode_stats, episode_metrics


def summarize_difficulty_rates(results_per_task, logger):
    logger.info("\n===============")
    logger.info("Success Rate by Difficulty:")
    difficulty_rates = {}

    for difficulty, tasks in DIFFICULTY_TO_TASKS.items():
        task_rates = [
            results_per_task.get(task, 0.0)
            for task in tasks
            if task in results_per_task
        ]
        if task_rates:
            avg_rate = sum(task_rates) / len(task_rates)
            difficulty_rates[difficulty] = avg_rate
            logger.info(
                f"{difficulty}: {avg_rate:.2%} "
                f"(averaged over {len(task_rates)} tasks)"
            )

    if difficulty_rates:
        overall_avg = sum(difficulty_rates.values()) / len(difficulty_rates)
        logger.info(
            f"\nOverall Average Success Rate (across all difficulties): {overall_avg:.2%}"
        )


def save_rollout_video(frames, out_path, video_temp_subsample):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        out_path,
        # frames 是 [T] 长度的图像列表，每个元素 shape=[H, W, 3]。
        # 这里按时间维抽帧，减少视频体积。
        [np.asarray(x) for x in frames[::video_temp_subsample]],
        fps=max(1, 25 // video_temp_subsample),
    )


def main(args):
    logger = setup_logger(args.exp_name, args.log_dir)
    policy = setup_policy(args)

    total_episodes = 0
    total_successes = 0
    total_all_infer_ms_sum = 0.0
    total_all_infer_count = 0
    total_guided_infer_ms_sum = 0.0
    total_guided_infer_count = 0
    total_bootstrap_infer_ms_sum = 0.0
    total_bootstrap_infer_count = 0
    total_episode_time_sum = 0.0
    total_success_time_sum = 0.0
    total_success_time_count = 0
    total_deadline_miss_count = 0

    results_per_task = {}
    task_time_metrics = {}

    for env_name in ENV_LIST:
        task_successes = 0
        task_all_infer_ms_sum = 0.0
        task_all_infer_count = 0
        task_guided_infer_ms_sum = 0.0
        task_guided_infer_count = 0
        task_bootstrap_infer_ms_sum = 0.0
        task_bootstrap_infer_count = 0
        task_episode_time_sum = 0.0
        task_success_time_sum = 0.0
        task_success_time_count = 0
        task_deadline_miss_count = 0

        for trial_id in range(args.num_trials_per_task):
            env = make_env(env_name)

            # 这里实例化的是 Algorithm 1 风格的 RTC 执行器：
            # 前台每个 control_dt 调 GetAction，后台常驻线程按 condition C 做 InferenceLoop。
            rtc_executor = AsyncRTCExecutor(
                policy=policy,
                action_chunk=args.action_chunk,
                control_dt=1.0 / args.control_hz,
                min_exec_horizon=args.rtc_min_exec_horizon,
                delay_buffer_size=args.rtc_delay_buffer_size,
                num_denoise_steps=args.num_steps,
                guidance_clip=args.rtc_guidance_clip,
                initial_delay_steps=args.rtc_initial_delay_steps,
                inject_delay_ms=args.inject_delay_ms,
                logger=logger,
                runtime_log=args.verbose_rtc,
            )

            success = 0
            frames = []
            episode_stats = []
            episode_metrics = None
            try:
                success, frames, episode_stats, episode_metrics = run_one_episode(
                    env=env,
                    env_name=env_name,
                    args=args,
                    logger=logger,
                    rtc_executor=rtc_executor,
                )
            finally:
                rtc_executor.close()
                executor_timing = rtc_executor.get_timing_summary()
                env.close()

            if episode_metrics is None:
                episode_metrics = {}
            episode_metrics.update(executor_timing)

            task_successes += success
            total_successes += success
            total_episodes += 1

            task_all_infer_ms_sum += episode_metrics["all_infer_ms_sum"]
            task_all_infer_count += episode_metrics["all_infer_count"]
            task_guided_infer_ms_sum += episode_metrics["guided_infer_ms_sum"]
            task_guided_infer_count += episode_metrics["guided_infer_count"]
            task_bootstrap_infer_ms_sum += episode_metrics["bootstrap_infer_ms_sum"]
            task_bootstrap_infer_count += episode_metrics["bootstrap_infer_count"]
            task_episode_time_sum += episode_metrics["episode_elapsed_s"]
            task_deadline_miss_count += episode_metrics["deadline_miss_count"]

            total_all_infer_ms_sum += episode_metrics["all_infer_ms_sum"]
            total_all_infer_count += episode_metrics["all_infer_count"]
            total_guided_infer_ms_sum += episode_metrics["guided_infer_ms_sum"]
            total_guided_infer_count += episode_metrics["guided_infer_count"]
            total_bootstrap_infer_ms_sum += episode_metrics["bootstrap_infer_ms_sum"]
            total_bootstrap_infer_count += episode_metrics["bootstrap_infer_count"]
            total_episode_time_sum += episode_metrics["episode_elapsed_s"]
            total_deadline_miss_count += episode_metrics["deadline_miss_count"]

            if episode_metrics["first_success_time_s"] is not None:
                task_success_time_sum += episode_metrics["first_success_time_s"]
                task_success_time_count += 1
                total_success_time_sum += episode_metrics["first_success_time_s"]
                total_success_time_count += 1

            if total_episodes <= args.num_save_videos:
                suffix = "success" if success else "failure"
                out_path = (
                    pathlib.Path(f"{args.log_dir}/{args.exp_name}/")
                    / f"{env_name}_{trial_id}_{suffix}.mp4"
                )
                save_rollout_video(frames, out_path, args.video_temp_subsample)

            # 可选: 把每一步的 RTC 统计量存成 json，后面便于分析 infer_ms / delay_steps。
            if args.save_step_stats:
                stats_dir = pathlib.Path(f"{args.log_dir}/{args.exp_name}/step_stats")
                stats_dir.mkdir(parents=True, exist_ok=True)
                stats_path = stats_dir / f"{env_name}_{trial_id}.json"
                write_json(stats_path, episode_stats)

            if args.save_episode_stats:
                stats_path = (
                    pathlib.Path(f"{args.log_dir}/{args.exp_name}/episode_stats")
                    / f"{env_name}_{trial_id}.json"
                )
                write_json(stats_path, episode_metrics)

        task_success_rate = task_successes / args.num_trials_per_task
        results_per_task[env_name] = task_success_rate

        task_summary = {
            "success_rate": task_success_rate,
            "avg_infer_ms_per_call": safe_div(task_all_infer_ms_sum, task_all_infer_count),
            "avg_guided_infer_ms_per_call": safe_div(
                task_guided_infer_ms_sum, task_guided_infer_count
            ),
            "avg_bootstrap_infer_ms": safe_div(
                task_bootstrap_infer_ms_sum, task_bootstrap_infer_count
            ),
            "avg_episode_time_s_all": safe_div(
                task_episode_time_sum, args.num_trials_per_task
            ),
            "avg_success_time_s": safe_div(
                task_success_time_sum, task_success_time_count
            ),
            "effective_time_per_success_s": safe_div(
                task_episode_time_sum, task_successes
            ),
            "avg_deadline_miss_per_episode": safe_div(
                task_deadline_miss_count, args.num_trials_per_task
            ),
            "infer_count": task_all_infer_count,
            "guided_infer_count": task_guided_infer_count,
            "bootstrap_infer_count": task_bootstrap_infer_count,
            "num_trials": args.num_trials_per_task,
            "num_successes": task_successes,
            "deadline_miss_count": task_deadline_miss_count,
        }
        task_time_metrics[env_name] = task_summary

        logger.info(
            f"Task: {env_name}, "
            f"Successes: {task_successes}/{args.num_trials_per_task}, "
            f"Success Rate: {task_success_rate:.2%}, "
            f"avg_infer_ms/call={format_metric(task_summary['avg_infer_ms_per_call'], 1)}, "
            f"avg_guided_infer_ms/call={format_metric(task_summary['avg_guided_infer_ms_per_call'], 1)}, "
            f"avg_bootstrap_infer_ms={format_metric(task_summary['avg_bootstrap_infer_ms'], 1)}, "
            f"avg_episode_time_s(all)={format_metric(task_summary['avg_episode_time_s_all'])}, "
            f"avg_success_time_s={format_metric(task_summary['avg_success_time_s'])}, "
            f"effective_time_per_success_s={format_metric(task_summary['effective_time_per_success_s'])}, "
            f"avg_deadline_miss/episode={format_metric(task_summary['avg_deadline_miss_per_episode'])}"
        )

    total_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    overall_summary = {
        "success_rate": total_success_rate,
        "avg_infer_ms_per_call": safe_div(total_all_infer_ms_sum, total_all_infer_count),
        "avg_guided_infer_ms_per_call": safe_div(
            total_guided_infer_ms_sum, total_guided_infer_count
        ),
        "avg_bootstrap_infer_ms": safe_div(
            total_bootstrap_infer_ms_sum, total_bootstrap_infer_count
        ),
        "avg_episode_time_s_all": safe_div(total_episode_time_sum, total_episodes),
        "avg_success_time_s": safe_div(
            total_success_time_sum, total_success_time_count
        ),
        "effective_time_per_success_s": safe_div(
            total_episode_time_sum, total_successes
        ),
        "avg_deadline_miss_per_episode": safe_div(
            total_deadline_miss_count, total_episodes
        ),
        "infer_count": total_all_infer_count,
        "guided_infer_count": total_guided_infer_count,
        "bootstrap_infer_count": total_bootstrap_infer_count,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "deadline_miss_count": total_deadline_miss_count,
    }

    logger.info(
        "[Overall] "
        f"success_rate={total_success_rate:.2%}, "
        f"avg_infer_ms/call={format_metric(overall_summary['avg_infer_ms_per_call'], 1)}, "
        f"avg_guided_infer_ms/call={format_metric(overall_summary['avg_guided_infer_ms_per_call'], 1)}, "
        f"avg_bootstrap_infer_ms={format_metric(overall_summary['avg_bootstrap_infer_ms'], 1)}, "
        f"avg_episode_time_s(all)={format_metric(overall_summary['avg_episode_time_s_all'])}, "
        f"avg_success_time_s={format_metric(overall_summary['avg_success_time_s'])}, "
        f"effective_time_per_success_s={format_metric(overall_summary['effective_time_per_success_s'])}, "
        f"avg_deadline_miss/episode={format_metric(overall_summary['avg_deadline_miss_per_episode'])}"
    )

    summary_path = pathlib.Path(f"{args.log_dir}/{args.exp_name}/timing_summary.json")
    write_json(
        summary_path,
        {
            "method": "rtc_async",
            "per_task": task_time_metrics,
            "overall": overall_summary,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 这一组参数与官方 metaworld_eval.py 基本保持一致。
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="metaworld_rtc",
        help="Experiment name used for naming log files and video save directories",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi0_metaworld",
        help="Config name, e.g. 'pi0_metaworld' or 'pi05_metaworld'",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained checkpoint dir or weights",
    )
    parser.add_argument(
        "--num_trials_per_task",
        type=int,
        default=10,
        help="Number of trials per task",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=160,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=5,
        help="Prediction horizon H used by RTC; for pi0_metaworld this is usually 5",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of policy sampling steps passed into setup_policy()",
    )
    parser.add_argument(
        "--num_save_videos",
        type=int,
        default=0,
        help="Only saves rollout videos for the first N episodes. Default is 0 to avoid I/O overhead",
    )
    parser.add_argument(
        "--video_temp_subsample",
        type=int,
        default=10,
        help="Save every Nth frame",
    )

    # 这一组参数专门控制 Algorithm 1 RTC 行为。
    parser.add_argument(
        "--control_hz",
        type=float,
        default=10.0,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--inject_delay_ms",
        type=float,
        default=0.0,
        help="Artificial delay injected into each inference call for testing RTC robustness",
    )
    parser.add_argument(
        "--rtc_min_exec_horizon",
        type=int,
        default=2,
        help="s_min in RTC: start background inference after at least this many executed actions",
    )
    parser.add_argument(
        "--rtc_delay_buffer_size",
        type=int,
        default=10,
        help="b in RTC: number of observed delay steps kept in the delay queue Q",
    )
    parser.add_argument(
        "--rtc_initial_delay_steps",
        type=int,
        default=1,
        help="d_init in RTC: initial delay estimate inserted into Q before the first inference",
    )
    parser.add_argument(
        "--rtc_guidance_clip",
        type=float,
        default=5.0,
        help="beta in RTC: maximum guidance weight used inside guided denoising",
    )

    # 下面这些旧参数保留为兼容项，当前 Algorithm 1 版本不再使用。
    parser.add_argument(
        "--replan_trigger_remaining",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--splice_overlap",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--commit_prefix_steps",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--conservative_delay_quantile",
        type=float,
        default=0.9,
        help=argparse.SUPPRESS,
    )

    # 其他控制项。
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=15,
        help="Initial dummy steps to let objects settle",
    )
    parser.add_argument(
        "--break_on_success",
        action="store_true",
        help="Break the episode immediately once success is achieved",
    )
    parser.add_argument(
        "--save_step_stats",
        action="store_true",
        help="Save per-step RTC stats as json",
    )
    parser.add_argument(
        "--save_episode_stats",
        action="store_true",
        help="Save per-episode timing stats as json",
    )
    parser.add_argument(
        "--verbose_rtc",
        action="store_true",
        help="Print detailed RTC stats every step",
    )
    args = parser.parse_args()
    main(args)
