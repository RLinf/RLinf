from omegaconf import DictConfig
from typing import TYPE_CHECKING, Optional
from rlinf.envs.adb.adb_env import ADBEnv
from rlinf.scheduler import Channel, Worker
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.data.datasets.android import AndroidWorldDataset
import sys
import os
import time

android_world_parent = "/mnt/project_rlinf/yuanqwang/mobile-agent/android_world"
if android_world_parent not in sys.path:
    sys.path.insert(0, android_world_parent)

from android_world.env import env_launcher
from android_world.episode_runner import EpisodeResult, _transpose_lod_to_dol
# ANSI color codes
# Note: Gpt4Wrapper is imported inside init_worker() to avoid import errors in main process
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"
COLOR_BOLD = "\033[1m"

def green(text: str) -> str:
    """Return text in green color."""
    return f"{COLOR_GREEN}{text}{COLOR_RESET}"

def yellow(text: str) -> str:
    """Return text in yellow color."""
    return f"{COLOR_YELLOW}{text}{COLOR_RESET}"

def blue(text: str) -> str:
    """Return text in blue color."""
    return f"{COLOR_BLUE}{text}{COLOR_RESET}"

def magenta(text: str) -> str:
    """Return text in magenta color."""
    return f"{COLOR_MAGENTA}{text}{COLOR_RESET}"

def cyan(text: str) -> str:
    """Return text in cyan color."""
    return f"{COLOR_CYAN}{text}{COLOR_RESET}"

def bold(text: str) -> str:
    """Return text in bold."""
    return f"{COLOR_BOLD}{text}{COLOR_RESET}"



if TYPE_CHECKING:
    from rlinf.scheduler.hardware import ADBHWInfo

class AndroidAgentWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.envs = []
        self.dataset: Optional[AndroidWorldDataset] = None
        self.current_env_idx = 0
        self.llm = None
        # 每个任务的单独计时：{task_idx: {metric_name: time}}
        self.per_task_timings: dict[int, dict[str, float]] = {}
        self.generate_input_channel = None
        self.generate_output_channel = None

        if not self.hardware_infos:
            raise ValueError(
                "InitWorker requires hardware_infos to bind ADB device(s). "
                "Please configure hardware in cluster.node_groups."
            )

    def init_with_channels(
        self,
        generate_input_channel,
        generate_output_channel
    ):
        self.generate_input_channel = generate_input_channel
        self.generate_output_channel = generate_output_channel

    def init_worker(self):
        # 使用 self.hardware_infos 获取当前 worker 绑定的硬件设备
        # 这样每个 worker 只负责一个硬件设备（一个 env）
        hardware_infos = self.hardware_infos
        self.log_info(f"Worker rank {self._rank} bound to {len(hardware_infos)} hardware device(s)")
        
        if len(hardware_infos) == 0:
            raise ValueError(f"Worker rank {self._rank} has no hardware device assigned")
        
        # 计时：env 创建（每个 worker 只创建一个 env）
        for idx, hw_info in enumerate(hardware_infos):
            hw_info: "ADBHWInfo"
            device_id = hw_info.config.device_id
            adb_path = hw_info.config.adb_path
            # 使用 worker rank 来区分 grpc_port，而不是 idx
            grpc_port = self.cfg.reward.get("grpc_port", 8554) + self._rank

            if ":" in device_id:
                console_port = int(device_id.split(":")[1]) - 1
            else:
                console_port = int(device_id.split("-")[1]) - 1

            with self.worker_timer(f"env_setup_device_{idx}"):
                self.log_info(
                    f"Loading and setting up env for device {device_id} on rank {self._rank}..."
                )
                env = env_launcher.load_and_setup_env(
                    console_port=console_port,
                    emulator_setup=False,
                    freeze_datetime=True,
                    adb_path=adb_path,
                    grpc_port=grpc_port,
                    device_id=device_id,
                )
                self.log_info(f"✓ AndroidWorld env created for device {device_id}")
                self.envs.append(env)
        self.log_info(f"Total envs created: {len(self.envs)} for worker rank {self._rank}")
        for i in range(len(self.envs)):
            self.log_info(f"  env[{i}]: device_id={hardware_infos[i].config.device_id}")

        # 计时：dataset 加载
        self.log_info("Loading AndroidWorld dataset...")
        with self.worker_timer("dataset_load"):
            tokenizer = hf_tokenizer(self.cfg.data.tokenizer.tokenizer_model)
            self.dataset = AndroidWorldDataset(
                config=self.cfg,
                tokenizer=tokenizer,
                seed=self.cfg.data.get("seed", 42),
            )
            
        self.log_info(
            f"✓ AndroidWorld dataset loaded: {len(self.dataset)} task instances"
        )

        # 延迟导入 Gpt4Wrapper，这样它会在 worker 进程中执行（使用正确的 Python 环境）
        from android_world.agents.infer import OpenAICompatibleWrapper
        from rlinf.workers.env.llm_wrapper import LLMWrapper
        model_name = os.environ.get('LOCAL_MODEL_NAME', 'Qwen3-VL-4B-Instruct')
        base_url = os.environ.get('LOCAL_API_BASE', 'http://localhost:22002/v1/chat/completions')
        api_key = os.environ.get('LOCAL_API_KEY', 'EMPTY')
        self.llm = OpenAICompatibleWrapper(model_name=model_name, base_url=base_url, api_key=api_key)
        # self.llm = LLMWrapper(
        #     generate_input_channel=self.generate_input_channel, 
        #     generate_output_channel=self.generate_output_channel
        #     )

    def process_task(self, task_idx: int, reward_worker_group_name: str = "RewardWorkerGroup"):
        """Process a single task from the dataset.
        
        Returns:
            dict with keys: task_idx, task_name, goal, complexity, reward, num_steps,
                            per_step_timings (list of per-step dicts), task_timings (aggregated)
        """
        self.per_task_timings[task_idx] = {}
        task_timings = self.per_task_timings[task_idx]
        
        task_item = self.dataset[task_idx]
        task = task_item.answer["task"]
        goal = task.goal
        meta = task_item.meta or {}
        task_name = meta.get('task_name', task_item.answer.get('task_name', 'unknown'))
        complexity = meta.get('complexity', None)
        user_prompt = meta.get('user_prompt', 'N/A')
        print(f"task name and params: {task_name}, user_prompt: {user_prompt}, task goal: {goal}")

        env, env_info = self._get_next_env() # get available env
        device_id = env_info['device_id']
        self.log_info(f"[Task {task_idx}] Selected device: {device_id}, worker rank: {self._rank}, current_env_idx after selection: {self.current_env_idx}")
        
        # 计时：任务初始化（同时记录到全局和当前任务）
        init_start = time.perf_counter()
        with self.worker_timer("task_initialize"):
            self.log_info(f"Running task {task} on device {device_id} (worker rank {self._rank})...")
            try:
                task.initialize_task(env)
            except Exception as e:
                # 在异常时输出设备信息，方便定位问题
                self.log_error(
                    f"Failed to initialize task {task_idx} on device {device_id} "
                    f"(worker rank {self._rank}): {type(e).__name__}: {e}"
                )
                raise
        init_duration = time.perf_counter() - init_start
        task_timings["task_initialize"] = init_duration

        # 合并 task 内部各阶段耗时到 task_timings，并打印
        init_breakdown = getattr(task, "initialization_timings", None) or {}
        for k, v in init_breakdown.items():
            task_timings[f"task_initialize_{k}"] = v
        if init_breakdown:
            base_total = sum(init_breakdown.values())
            subclass_duration = init_duration - base_total
            task_timings["task_initialize_subclass"] = max(0.0, subclass_duration)
            self.log_info(
                f"  Task init 细分: total={init_duration:.2f}s | "
                + " | ".join(f"{k}={v:.2f}s" for k, v in sorted(init_breakdown.items()))
                + (f" | subclass={subclass_duration:.2f}s" if subclass_duration > 0 else "")
            )

        # 计时：agent 执行（同时记录到全局和当前任务）
        agent_start = time.perf_counter()
        with self.worker_timer("agent_run"):
            agent_result = self._run_agent(env, complexity, task, goal, task_idx)
        agent_duration = time.perf_counter() - agent_start
        task_timings["agent_run"] = agent_duration
        env_info["agent_result"] = agent_result
        env_info["task_item"] = task_item
        env_info["task_idx"] = task_idx  # 添加 task_idx 供 RewardWorker 使用

        # 细粒度计时：send 通信时间（同时记录到全局和当前任务）
        send_start = time.perf_counter()
        self.send(
            object=env_info,
            dst_group_name=reward_worker_group_name,
            dst_rank=self._rank,
        )
        send_duration = time.perf_counter() - send_start
        self._timer_metrics["send_env_info"] = (
            self._timer_metrics.get("send_env_info", 0.0) + send_duration
        )
        task_timings["send_env_info"] = send_duration
        self.log_info(f"Sent env info[{task}] to {reward_worker_group_name}[{self._rank}]")

        # 细粒度计时：recv 通信时间（同时记录到全局和当前任务）
        recv_start = time.perf_counter()
        reward = self.recv(
            src_group_name=reward_worker_group_name,
            src_rank=self._rank,
        )
        recv_duration = time.perf_counter() - recv_start
        self._timer_metrics["recv_reward"] = (
            self._timer_metrics.get("recv_reward", 0.0) + recv_duration
        )
        task_timings["recv_reward"] = recv_duration
        # 使用绿色显示奖励信息，突出显示
        reward_msg = f"Received reward from {reward_worker_group_name}[{self._rank}], reward: {bold(green(str(reward)))}"
        self.log_info(reward_msg)

        # 计时：任务清理（同时记录到全局和当前任务）
        tear_down_start = time.perf_counter()
        with self.worker_timer("task_tear_down"):
            task.tear_down(env)
        tear_down_duration = time.perf_counter() - tear_down_start
        task_timings["task_tear_down"] = tear_down_duration
        self.log_info(f"Torn down env for task {task_idx}")

        # ========== 任务计时汇总 ==========
        pure_inference_s = task_timings.get("llm_pure_inference_total", 0.0)
        llm_total_s = task_timings.get("llm_inference_total", 0.0)
        img_encode_s = task_timings.get("image_encode_total", 0.0)
        vision_preproc_s = task_timings.get("vision_preprocess_total", 0.0)
        agent_put_s = task_timings.get("llm_agent_put_total", 0.0)
        agent_get_s = task_timings.get("llm_agent_get_total", 0.0)
        rlinf_overhead_s = task_timings.get("llm_rlinf_overhead_total", 0.0)
        num_steps = task_timings.get("num_steps", 0)
        agent_run_s = task_timings.get("agent_step_total", 0.0)
        llm_pct = f"{llm_total_s / agent_run_s * 100:.1f}%" if agent_run_s > 0 else "N/A"
        pure_pct = f"{pure_inference_s / agent_run_s * 100:.1f}%" if agent_run_s > 0 else "N/A"
        self.log_info(
            f"\n{'='*60}\n"
            f"  Task {task_idx} LLM 计时汇总 ({num_steps} steps)\n"
            f"{'='*60}\n"
            f"  ★ 纯文本推理时间 (engine inference):  {pure_inference_s:.2f}s\n"
            f"  ★ LLM 总时间 (agent predict_mm):      {llm_total_s:.2f}s\n"
            f"{'─'*60}\n"
            f"  LLM 时间分解:\n"
            f"    图像base64编码 (agent侧):           {img_encode_s:.2f}s\n"
            f"    channel put (agent→engine):          {agent_put_s:.2f}s\n"
            f"    vision预处理 (engine侧):            {vision_preproc_s:.2f}s\n"
            f"    纯文本推理 (model prefill+decode):   {pure_inference_s:.2f}s\n"
            f"    channel get 等待:                    {agent_get_s:.2f}s\n"
            f"    RLinf 通信/调度开销:                 {rlinf_overhead_s:.2f}s\n"
            f"{'─'*60}\n"
            f"  Agent 总运行时间:                      {agent_run_s:.2f}s\n"
            f"  LLM 占比:                              {llm_pct}\n"
            f"  纯推理占比:                            {pure_pct}\n"
            f"{'='*60}"
        )

        return {
            "task_idx": task_idx,
            "task_name": task_name,
            "goal": goal,
            "complexity": complexity,
            "reward": reward,
            "num_steps": task_timings.get("num_steps", 0),
            "per_step_timings": task_timings.get("_per_step_timings", []),
            "task_timings": {k: v for k, v in task_timings.items() if k != "_per_step_timings"},
        }

    def run_task_list(
        self,
        task_indices: list[int],
        reward_worker_group_name: str = "RewardWorkerGroup",
    ):
        """在本 worker 上顺序跑一串任务，内部仍用 process_task + RewardWorker 协议。"""
        results = []
        for task_idx in task_indices:
            try:
                result = self.process_task(
                    task_idx=task_idx,
                    reward_worker_group_name=reward_worker_group_name,
                )
                results.append(result)
            except Exception as e:
                self.log_error(f"Task {task_idx} failed: {e}")
                results.append({
                    "task_idx": task_idx,
                    "task_name": "ERROR",
                    "goal": "",
                    "complexity": None,
                    "reward": 0.0,
                    "num_steps": 0,
                    "per_step_timings": [],
                    "task_timings": {},
                    "error": str(e),
                })
        return results

    def _run_agent(self, env, complexity, task, goal, task_idx: int):
        from android_world.agents import m3a
        from android_world.env import adb_utils, representation_utils
        max_n_steps = 25 #int(complexity * 10)
        agent = m3a.M3A(env, self.llm)
        agent.reset(go_home_on_reset=False)

        agent.set_max_steps(max_n_steps)
        if hasattr(task, 'guidelines'):
            agent.set_task_guidelines(task.guidelines)
        actions_output = []
        task_timings = self.per_task_timings[task_idx]
        # 初始化细粒度计时指标
        task_timings["agent_step_total"] = 0.0      # 每个 step 的总时间
        task_timings["llm_inference_total"] = 0.0    # LLM 推理总时间（action selection + summary）
        task_timings["llm_action_select"] = 0.0      # LLM 动作选择推理时间
        task_timings["llm_summarize"] = 0.0          # LLM 总结推理时间
        task_timings["llm_agent_put_total"] = 0.0    # Agent 侧 channel put 总时间（RLinf 传输）
        task_timings["llm_agent_get_total"] = 0.0    # Agent 侧 channel get 总时间（含等待 engine）
        task_timings["llm_engine_inference_total"] = 0.0  # Engine 侧纯生成时间（从 result 带回）
        task_timings["llm_pure_inference_total"] = 0.0     # 与 HTTP 可比的纯 LLM 推理时间（= engine 侧 async_generate 耗时）
        task_timings["llm_rlinf_overhead_total"] = 0.0   # get 内除 engine 推理外的 RLinf 传输/调度时间
        task_timings["image_encode_total"] = 0.0         # Agent 侧图像 base64 编码时间
        task_timings["vision_preprocess_total"] = 0.0    # Engine 侧 vision 预处理时间（process_vision_info + chat_template）
        task_timings["get_state_total"] = 0.0        # get_state 总时间（截图 + UI 树）
        task_timings["get_state_before_total"] = 0.0 # 每步开始前的独立 get_state 时间
        task_timings["get_state_after_total"] = 0.0  # 每步动作后的独立 get_state 时间
        task_timings["execute_action_total"] = 0.0   # execute_action 总时间（含内部 get_state）
        task_timings["execute_adb_action"] = 0.0     # 纯 ADB 命令时间（execute_action 中减去内部 get_state）
        task_timings["transition_pause_total"] = 0.0 # step 开始前的 transition_pause 等待
        task_timings["wait_after_action_total"] = 0.0 # 动作后的 wait_after_action sleep
        task_timings["som_annotation_total"] = 0.0   # SOM 标注（图像处理）时间
        # get_state 内部拆分
        task_timings["gs_env_step_total"] = 0.0           # controller._env.step (gRPC 通信) 总时间
        task_timings["gs_uiautomator_dump_total"] = 0.0   # uiautomator_dump 总时间
        task_timings["gs_xml_parse_total"] = 0.0          # XML 解析为 UIElements 总时间
        task_timings["gs_glue_total"] = 0.0               # get_state 中的 Python glue 开销
        task_timings["_per_step_timings"] = []               # 每步详细计时列表
        actual_steps = 0
        
        # ========== 累加器（闭包中可变） ==========
        accum = {
            "llm": 0.0,                # 所有 LLM 调用累计
            "llm_call_count": 0,       # LLM 调用次数（奇数=action, 偶数=summary）
            "llm_action_select": 0.0,   # LLM 动作选择累计
            "llm_summarize": 0.0,      # LLM 总结累计
            "llm_agent_put": 0.0,      # Agent 侧 put 到 channel 累计（RLinf 传输）
            "llm_agent_get": 0.0,      # Agent 侧 get 从 channel 累计（含等待 engine）
            "llm_engine_inference": 0.0,  # Engine 侧纯生成时间累计（从 result 带回）
            "image_encode": 0.0,         # Agent 侧图像 base64 编码时间累计
            "vision_preprocess": 0.0,    # Engine 侧 vision 预处理时间累计
            "get_state": 0.0,           # 独立 get_state 累计（before+after）
            "get_state_before": 0.0,    # 每步开始前的独立 get_state 累计
            "get_state_after": 0.0,     # 每步动作后的独立 get_state 累计
            "execute_action": 0.0,      # execute_action 整体累计
            "exec_inner_get_state": 0.0, # execute_action 内部的 get_state
            "transition_pause": 0.0,    # transition_pause 累计
            "wait_after_action": 0.0,        # wait_after_action 累计
            "som_annotation": 0.0,           # SOM 标注累计
            # get_state 内部拆分
            "gs_env_step": 0.0,              # controller._env.step (gRPC) 累计
            "gs_uiautomator_dump": 0.0,      # uiautomator_dump 累计
            "gs_xml_parse": 0.0,             # xml_dump_to_ui_elements 累计
            # token 统计
            "llm_prompt_tokens": 0,                # 所有调用的 prompt_tokens 累计
            "llm_completion_tokens": 0,           # 所有调用的 completion_tokens 累计
            "llm_action_prompt_tokens": 0,        # 动作选择调用的 prompt_tokens 累计
            "llm_action_completion_tokens": 0,    # 动作选择调用的 completion_tokens 累计
            "llm_summary_prompt_tokens": 0,       # 总结调用的 prompt_tokens 累计
            "llm_summary_completion_tokens": 0,   # 总结调用的 completion_tokens 累计
        }
        inside_execute_action = {"flag": False}
        inside_step = {"flag": False}  # 标记是否在 agent.step 内部
        # 统计“独立 get_state”在每个 step 中的调用顺序：0 = before, 1 = after
        independent_gs_counter = {"value": 0}
        
        # ========== 包装 LLM predict_mm ==========
        original_predict_mm = agent.llm.predict_mm
        def timed_predict_mm(*args, **kwargs):
            llm_start = time.perf_counter()
            result = original_predict_mm(*args, **kwargs)
            llm_duration = time.perf_counter() - llm_start
            accum["llm"] += llm_duration
            accum["llm_call_count"] += 1
            if accum["llm_call_count"] % 2 == 1:
                accum["llm_action_select"] += llm_duration
            else:
                accum["llm_summarize"] += llm_duration
            # 从 result 中取出 LLM 细粒度计时（LLMWrapper / engine 写入）
            if isinstance(result, (tuple, list)) and len(result) >= 3:
                extra = result[2]
                if isinstance(extra, dict):
                    accum["llm_agent_put"] += extra.get("agent_put_s", 0.0)
                    accum["llm_agent_get"] += extra.get("agent_get_s", 0.0)
                    accum["llm_engine_inference"] += extra.get("engine_inference_s", 0.0)
                    accum["image_encode"] += extra.get("image_encode_s", 0.0)
                    accum["vision_preprocess"] += extra.get("vision_preprocess_s", 0.0)

                    # token 数量（Engine / HTTP wrapper 需在 extra 中填入 prompt_tokens / completion_tokens）
                    pt = int(extra.get("prompt_tokens", 0) or 0)
                    ct = int(extra.get("completion_tokens", 0) or 0)
                    accum["llm_prompt_tokens"] += pt
                    accum["llm_completion_tokens"] += ct
                    if accum["llm_call_count"] % 2 == 1:
                        accum["llm_action_prompt_tokens"] += pt
                        accum["llm_action_completion_tokens"] += ct
                    else:
                        accum["llm_summary_prompt_tokens"] += pt
                        accum["llm_summary_completion_tokens"] += ct

                    call_type = "action" if accum["llm_call_count"] % 2 == 1 else "summary"
                    self.log_info(
                        f"[LLM token diag] call#{accum['llm_call_count']} ({call_type}) "
                        f"prompt_tokens={pt} "
                        f"completion_tokens={ct} "
                        f"engine_inference={extra.get('engine_inference_s', 0):.3f}s "
                        f"llm_wall={llm_duration:.3f}s"
                    )
            return result
        
        # ========== 包装 env.get_state ==========
        original_get_state = env.get_state
        def timed_get_state(*args, **kwargs):
            gs_start = time.perf_counter()
            result = original_get_state(*args, **kwargs)
            gs_duration = time.perf_counter() - gs_start
            if inside_execute_action["flag"]:
                # execute_action 内部的 get_state，记录到子项
                accum["exec_inner_get_state"] += gs_duration
            else:
                # 独立 get_state（step 开始获取状态 / 动作后获取状态）
                accum["get_state"] += gs_duration
                order = independent_gs_counter["value"]
                if order == 0:
                    accum["get_state_before"] += gs_duration
                elif order == 1:
                    accum["get_state_after"] += gs_duration
                # 超过 2 次的独立 get_state 仍然计入 get_state，但不再细分
                independent_gs_counter["value"] = order + 1
            return result
        
        # ========== 包装 env.execute_action ==========
        original_execute_action = env.execute_action
        def timed_execute_action(*args, **kwargs):
            ea_start = time.perf_counter()
            inside_execute_action["flag"] = True
            try:
                result = original_execute_action(*args, **kwargs)
            finally:
                inside_execute_action["flag"] = False
            ea_duration = time.perf_counter() - ea_start
            accum["execute_action"] += ea_duration
            return result
        
        # ========== 包装 time.sleep（拦截 M3A 中的 sleep） ==========
        import android_world.agents.m3a as m3a_module
        import android_world.agents.base_agent as base_agent_module
        original_time_sleep = time.sleep
        def timed_sleep(seconds):
            if not inside_step["flag"]:
                return original_time_sleep(seconds)
            sleep_start = time.perf_counter()
            original_time_sleep(seconds)
            sleep_duration = time.perf_counter() - sleep_start
            # 区分不同的 sleep：
            # wait_after_action_seconds = 2.0（M3A step 中动作后的 sleep）
            # transition_pause = 1.0（get_post_transition_state 中的 sleep）
            # actuation 中的 sleep(1.0) 归到 execute_action（已在 timed_execute_action 中计时）
            if inside_execute_action["flag"]:
                pass  # 已包含在 execute_action 的计时中
            elif abs(seconds - agent.wait_after_action_seconds) < 0.01:
                accum["wait_after_action"] += sleep_duration
            elif agent._transition_pause is not None and abs(seconds - agent._transition_pause) < 0.01:
                accum["transition_pause"] += sleep_duration
        
        # ========== 包装 SOM 标注（m3a_utils.add_ui_element_mark） ==========
        from android_world.agents import m3a_utils
        original_add_ui_element_mark = m3a_utils.add_ui_element_mark
        def timed_add_ui_element_mark(*args, **kwargs):
            som_start = time.perf_counter()
            result = original_add_ui_element_mark(*args, **kwargs)
            som_duration = time.perf_counter() - som_start
            accum["som_annotation"] += som_duration
            return result

        # ========== 包装 controller._env.step (gRPC) / uiautomator_dump / xml_parse ==========
        controller = env.controller
        original_env_step = controller._env.step

        def timed_env_step(*args, **kwargs):
            es_start = time.perf_counter()
            result = original_env_step(*args, **kwargs)
            es_duration = time.perf_counter() - es_start
            accum["gs_env_step"] += es_duration
            return result

        original_uiautomator_dump = adb_utils.uiautomator_dump

        def timed_uiautomator_dump(*args, **kwargs):
            ud_start = time.perf_counter()
            result = original_uiautomator_dump(*args, **kwargs)
            ud_duration = time.perf_counter() - ud_start
            accum["gs_uiautomator_dump"] += ud_duration
            return result

        original_xml_dump_to_ui_elements = representation_utils.xml_dump_to_ui_elements

        def timed_xml_dump_to_ui_elements(*args, **kwargs):
            xp_start = time.perf_counter()
            result = original_xml_dump_to_ui_elements(*args, **kwargs)
            xp_duration = time.perf_counter() - xp_start
            accum["gs_xml_parse"] += xp_duration
            return result
        
        # ========== 应用 monkey patching ==========
        agent.llm.predict_mm = timed_predict_mm
        env.execute_action = timed_execute_action
        env.get_state = timed_get_state
        m3a_utils.add_ui_element_mark = timed_add_ui_element_mark
        time.sleep = timed_sleep
        controller._env.step = timed_env_step
        adb_utils.uiautomator_dump = timed_uiautomator_dump
        representation_utils.xml_dump_to_ui_elements = timed_xml_dump_to_ui_elements
        
        try:
            for i in range(max_n_steps):
                # 记录 step 开始前的各累加器快照，并重置本步的独立 get_state 计数
                snap = {k: v for k, v in accum.items()}
                independent_gs_counter["value"] = 0
                
                step_start = time.perf_counter()
                inside_step["flag"] = True
                result = agent.step(goal)
                inside_step["flag"] = False
                step_duration = time.perf_counter() - step_start
                
                # 计算当前 step 的各项增量
                d_llm = accum["llm"] - snap["llm"]
                d_llm_action = accum["llm_action_select"] - snap["llm_action_select"]
                d_llm_summary = accum["llm_summarize"] - snap["llm_summarize"]
                d_llm_agent_put = accum["llm_agent_put"] - snap["llm_agent_put"]
                d_llm_agent_get = accum["llm_agent_get"] - snap["llm_agent_get"]
                d_llm_engine_inference = accum["llm_engine_inference"] - snap["llm_engine_inference"]
                d_image_encode = accum["image_encode"] - snap["image_encode"]
                d_vision_preprocess = accum["vision_preprocess"] - snap["vision_preprocess"]
                d_get_state = accum["get_state"] - snap["get_state"]
                d_get_state_before = accum["get_state_before"] - snap["get_state_before"]
                d_get_state_after = accum["get_state_after"] - snap["get_state_after"]
                d_exec_action = accum["execute_action"] - snap["execute_action"]
                d_exec_inner_gs = accum["exec_inner_get_state"] - snap["exec_inner_get_state"]
                d_transition = accum["transition_pause"] - snap["transition_pause"]
                d_wait_after = accum["wait_after_action"] - snap["wait_after_action"]
                d_som = accum["som_annotation"] - snap["som_annotation"]
                d_gs_env_step = accum["gs_env_step"] - snap["gs_env_step"]
                d_gs_uia = accum["gs_uiautomator_dump"] - snap["gs_uiautomator_dump"]
                d_gs_xml = accum["gs_xml_parse"] - snap["gs_xml_parse"]
                d_gs_glue = max(0.0, d_get_state - d_gs_env_step - d_gs_uia - d_gs_xml)
                # 纯 ADB 命令 = execute_action 整体 - 内部 get_state
                d_pure_adb = d_exec_action - d_exec_inner_gs
                # LLM 时间关系：LLM_total = agent_put + agent_get；agent_get 内包含 engine 推理，故 rlinf_overhead = get - engine_infer
                d_llm_rlinf_overhead = max(0.0, d_llm_agent_get - d_llm_engine_inference)
                # 一致性校验：LLM 总时间应等于 put + get（允许小误差）
                llm_put_get = d_llm_agent_put + d_llm_agent_get
                if abs(d_llm - llm_put_get) > 0.05:
                    self.log_info(
                        f"[LLM timing sanity] step {i} d_llm={d_llm:.3f}s vs put+get={llm_put_get:.3f}s (diff={d_llm - llm_put_get:.3f}s)"
                    )
                # 其他未归类时间
                d_other = step_duration - d_llm - d_get_state - d_exec_action - d_transition - d_wait_after - d_som

                # token 增量（该 step 内）
                d_prompt_tokens = accum["llm_prompt_tokens"] - snap.get("llm_prompt_tokens", 0)
                d_completion_tokens = accum["llm_completion_tokens"] - snap.get("llm_completion_tokens", 0)
                d_action_prompt_tokens = accum["llm_action_prompt_tokens"] - snap.get("llm_action_prompt_tokens", 0)
                d_action_completion_tokens = accum["llm_action_completion_tokens"] - snap.get("llm_action_completion_tokens", 0)
                d_summary_prompt_tokens = accum["llm_summary_prompt_tokens"] - snap.get("llm_summary_prompt_tokens", 0)
                d_summary_completion_tokens = accum["llm_summary_completion_tokens"] - snap.get("llm_summary_completion_tokens", 0)
                
                # 累计到全局 _timer_metrics
                metrics = self._timer_metrics
                metrics["agent_step_total"] = metrics.get("agent_step_total", 0.0) + step_duration
                metrics["llm_inference_total"] = metrics.get("llm_inference_total", 0.0) + d_llm
                metrics["llm_action_select"] = metrics.get("llm_action_select", 0.0) + d_llm_action
                metrics["llm_summarize"] = metrics.get("llm_summarize", 0.0) + d_llm_summary
                metrics["llm_agent_put_total"] = metrics.get("llm_agent_put_total", 0.0) + d_llm_agent_put
                metrics["llm_agent_get_total"] = metrics.get("llm_agent_get_total", 0.0) + d_llm_agent_get
                metrics["llm_engine_inference_total"] = metrics.get("llm_engine_inference_total", 0.0) + d_llm_engine_inference
                metrics["llm_pure_inference_total"] = metrics.get("llm_pure_inference_total", 0.0) + d_llm_engine_inference
                metrics["llm_rlinf_overhead_total"] = metrics.get("llm_rlinf_overhead_total", 0.0) + d_llm_rlinf_overhead
                metrics["image_encode_total"] = metrics.get("image_encode_total", 0.0) + d_image_encode
                metrics["vision_preprocess_total"] = metrics.get("vision_preprocess_total", 0.0) + d_vision_preprocess
                metrics["get_state_total"] = metrics.get("get_state_total", 0.0) + d_get_state
                metrics["get_state_before_total"] = metrics.get("get_state_before_total", 0.0) + d_get_state_before
                metrics["get_state_after_total"] = metrics.get("get_state_after_total", 0.0) + d_get_state_after
                metrics["execute_action_total"] = metrics.get("execute_action_total", 0.0) + d_exec_action
                metrics["execute_adb_action"] = metrics.get("execute_adb_action", 0.0) + d_pure_adb
                metrics["transition_pause_total"] = metrics.get("transition_pause_total", 0.0) + d_transition
                metrics["wait_after_action_total"] = metrics.get("wait_after_action_total", 0.0) + d_wait_after
                metrics["som_annotation_total"] = metrics.get("som_annotation_total", 0.0) + d_som
                metrics["gs_env_step_total"] = metrics.get("gs_env_step_total", 0.0) + d_gs_env_step
                metrics["gs_uiautomator_dump_total"] = metrics.get("gs_uiautomator_dump_total", 0.0) + d_gs_uia
                metrics["gs_xml_parse_total"] = metrics.get("gs_xml_parse_total", 0.0) + d_gs_xml
                metrics["gs_glue_total"] = metrics.get("gs_glue_total", 0.0) + d_gs_glue
                
                # 累计到当前任务计时
                task_timings["agent_step_total"] += step_duration
                task_timings["llm_inference_total"] += d_llm
                task_timings["llm_action_select"] += d_llm_action
                task_timings["llm_summarize"] += d_llm_summary
                task_timings["llm_agent_put_total"] += d_llm_agent_put
                task_timings["llm_agent_get_total"] += d_llm_agent_get
                task_timings["llm_engine_inference_total"] += d_llm_engine_inference
                task_timings["llm_pure_inference_total"] += d_llm_engine_inference  # 与 HTTP 可比的纯推理
                task_timings["llm_rlinf_overhead_total"] += d_llm_rlinf_overhead
                task_timings["image_encode_total"] += d_image_encode
                task_timings["vision_preprocess_total"] += d_vision_preprocess
                task_timings["get_state_total"] += d_get_state
                task_timings["get_state_before_total"] += d_get_state_before
                task_timings["get_state_after_total"] += d_get_state_after
                task_timings["execute_action_total"] += d_exec_action
                task_timings["execute_adb_action"] += d_pure_adb
                task_timings["transition_pause_total"] += d_transition
                task_timings["wait_after_action_total"] += d_wait_after
                task_timings["som_annotation_total"] += d_som
                 # 细分 get_state 内部组成
                task_timings["gs_env_step_total"] += d_gs_env_step
                task_timings["gs_uiautomator_dump_total"] += d_gs_uia
                task_timings["gs_xml_parse_total"] += d_gs_xml
                task_timings["gs_glue_total"] += d_gs_glue
                
                actual_steps += 1
                # 保存每步详细计时和 token 数
                task_timings["_per_step_timings"].append({
                    "step": i,
                    "total": round(step_duration, 3),
                    "llm": round(d_llm, 3),
                    "llm_action": round(d_llm_action, 3),
                    "llm_summary": round(d_llm_summary, 3),
                    "llm_agent_put": round(d_llm_agent_put, 3),
                    "llm_agent_get": round(d_llm_agent_get, 3),
                    "llm_engine_inference": round(d_llm_engine_inference, 3),
                    "llm_pure_inference": round(d_llm_engine_inference, 3),
                    "llm_rlinf_overhead": round(d_llm_rlinf_overhead, 3),
                    "image_encode": round(d_image_encode, 3),
                    "vision_preprocess": round(d_vision_preprocess, 3),
                    "get_state": round(d_get_state, 3),
                    "get_state_before": round(d_get_state_before, 3),
                    "get_state_after": round(d_get_state_after, 3),
                    "gs_env_step": round(d_gs_env_step, 3),
                    "gs_uia": round(d_gs_uia, 3),
                    "gs_xml": round(d_gs_xml, 3),
                    "gs_glue": round(d_gs_glue, 3),
                    "exec_action": round(d_exec_action, 3),
                    "adb_cmd": round(d_pure_adb, 3),
                    "inner_gs": round(d_exec_inner_gs, 3),
                    "transition_pause": round(d_transition, 3),
                    "wait_after": round(d_wait_after, 3),
                    "som": round(d_som, 3),
                    "other": round(d_other, 3),
                    # token 统计（每 step）
                    "llm_prompt_tokens": d_prompt_tokens,
                    "llm_completion_tokens": d_completion_tokens,
                    "llm_action_prompt_tokens": d_action_prompt_tokens,
                    "llm_action_completion_tokens": d_action_completion_tokens,
                    "llm_summary_prompt_tokens": d_summary_prompt_tokens,
                    "llm_summary_completion_tokens": d_summary_completion_tokens,
                })
                # 纯文本推理 = engine_inference（模型 prefill + decode）
                # LLM总时间 = predict_mm 整体（含图像编码 + channel通信 + vision预处理 + 推理）
                self.log_info(
                    f"Step {i} | total: {step_duration:.2f}s | "
                    f"★纯文本推理: {d_llm_engine_inference:.2f}s | "
                    f"★LLM总时间: {d_llm:.2f}s "
                    f"[img_encode: {d_image_encode:.2f}s, vision_preproc: {d_vision_preprocess:.2f}s, "
                    f"put: {d_llm_agent_put:.2f}s, get: {d_llm_agent_get:.2f}s, rlinf开销: {d_llm_rlinf_overhead:.2f}s] | "
                    f"LLM细分(action: {d_llm_action:.2f}s, summary: {d_llm_summary:.2f}s) | "
                    f"get_state: {d_get_state:.2f}s "
                    f"(before: {d_get_state_before:.2f}s, after: {d_get_state_after:.2f}s | "
                    f"env_step: {d_gs_env_step:.2f}s, uia: {d_gs_uia:.2f}s, xml: {d_gs_xml:.2f}s, glue: {d_gs_glue:.2f}s) | "
                    f"exec_action: {d_exec_action:.2f}s (adb_cmd: {d_pure_adb:.2f}s, inner_gs: {d_exec_inner_gs:.2f}s) | "
                    f"transition_pause: {d_transition:.2f}s | wait_after: {d_wait_after:.2f}s | "
                    f"SOM: {d_som:.3f}s | other: {d_other:.2f}s"
                )
                actions_output.append(result.data)
                print(f"=======================step{i}========================")
                print(f"action: {result.data['action_output_json']} \n  reasoning: {result.data['action_reason']} \n summary: {result.data['summary']} \n\n")
                if result.done:
                    task_timings["num_steps"] = actual_steps
                    return EpisodeResult(
                        done=result.done,
                        step_data=_transpose_lod_to_dol(actions_output),
                    )
        finally:
            # 恢复所有 monkey patching
            inside_step["flag"] = False
            agent.llm.predict_mm = original_predict_mm
            env.execute_action = original_execute_action
            env.get_state = original_get_state
            m3a_utils.add_ui_element_mark = original_add_ui_element_mark
            time.sleep = original_time_sleep
            controller._env.step = original_env_step
            adb_utils.uiautomator_dump = original_uiautomator_dump
            representation_utils.xml_dump_to_ui_elements = original_xml_dump_to_ui_elements
        
        # 达到最大步数
        task_timings["num_steps"] = actual_steps
        self.log_info(f"Task {task} reached max steps {max_n_steps} without completing.")
        return EpisodeResult(
            done=False,
            step_data=_transpose_lod_to_dol(actions_output),
        )

    def _get_next_env(self):
        # 每个 worker 只有一个 env，直接返回
        if len(self.envs) == 0:
            raise ValueError(f"Worker rank {self._rank} has no env initialized")
        
        env = self.envs[0]  # 每个 worker 只有一个 env
        
        # 使用 self.hardware_infos 获取当前 worker 绑定的硬件信息
        hardware_infos = self.hardware_infos
        if len(hardware_infos) == 0:
            raise ValueError(f"Worker rank {self._rank} has no hardware device assigned")
        
        hw_info = hardware_infos[0]  # 每个 worker 只有一个硬件设备
        hw_info: "ADBHWInfo"
        device_id = hw_info.config.device_id
        adb_path = hw_info.config.adb_path
        # 使用 worker rank 来区分 grpc_port
        grpc_port = self.cfg.reward.get("grpc_port", 8554) + self._rank
        
        if ":" in device_id:
            console_port = int(device_id.split(":")[1]) - 1
        else:
            console_port = int(device_id.split("-")[1]) - 1
        
        env_info = {
            "device_id": device_id,
            "adb_path": adb_path,
            "grpc_port": grpc_port,
            "console_port": console_port,
        }
        
        self.log_info(
            f"Using env[0] (device: {device_id}, grpc_port: {grpc_port}) "
            f"on worker rank {self._rank}"
        )
        return env, env_info

    def get_dataset_size(self):
        """返回 dataset 中的任务总数."""
        return len(self.dataset) if self.dataset else 0

    def get_timings(self):
        """返回当前 worker 已记录的计时信息（单位：秒）.
        
        Returns:
            dict: 包含 'per_task' 和 'total' 两个键
                - 'per_task': {task_idx: {metric_name: time}} 每个任务的计时
                - 'total': {metric_name: total_time} 所有任务的汇总时间
        """
        return {
            "per_task": dict(self.per_task_timings),
            "total": dict(self._timer_metrics),
        }
