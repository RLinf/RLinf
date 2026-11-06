import os, sys, torch, torch, numpy as np
B="/mnt/public/zhangyuzhuo"; R=f"{B}/work/RLinf_v2"
os.environ.update({
  "PYTHONPATH": R, "EMBODIED_PATH": f"{R}/examples/embodiment", "REPO_PATH": R,
  "LINGBOT_VLA_V2_PATH": f"{B}/venvs/rlinf_v2/lingbot-vla-v2",
  "LINGBOT_VLA_V2_MODEL_ROOT": f"{B}/models/lingbot-vla-v2-6b-robotwin",
  "LINGBOT_VLA_V2_CHECKPOINT": f"{B}/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt",
  "QWEN3VL_PATH": f"{B}/models/Qwen3-VL-4B-Instruct",
  "ROBOTWIN_PATH": f"{B}/robotwin/RoboTwin", "ROBOTWIN_ASSETS_PATH": f"{B}/robotwin/RoboTwin",
  "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "CUDA_VISIBLE_DEVICES": "0"})
sys.path.insert(0, R); sys.path.insert(0, f"{B}/robotwin/RoboTwin"); os.chdir(f"{R}/evaluations")

try:
    import torch as _t
    _frac = float(os.environ.get("MAX_GPU_FRAC", "0.25"))
    _t.cuda.set_per_process_memory_fraction(_frac, 0)
    print(f"[限制] 本进程显存上限 = {_frac*100:.0f}%")
except Exception as _e:
    print("[限制] 设置失败:", _e)

from hydra import initialize, compose
with initialize(config_path="robotwin", version_base=None):
    cfg = compose(config_name="robotwin_click_bell_lingbotvlav2_eval", overrides=sys.argv[1:])
from rlinf.envs import get_env_cls
env = get_env_cls(cfg.env.eval.env_type, cfg.env.eval)(cfg=cfg.env.eval, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
obs, info = env.reset()
print("reset state:", obs["states"][0][:7].tolist())
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="click_bell_clean_50", root=f"{B}/datasets/robotwin2_lerobot/click_bell_clean_50")
frames = list(range(81))
acts = torch.stack([ds[i]["action"].float() for i in frames])   # 单个 episode 的 81 帧
print("GT 动作 shape:", tuple(acts.shape), "范围:", float(acts.min()), float(acts.max()))
ok = False
for c in range(81):
    chunk = acts[c:c+1].unsqueeze(0)   # 逐步执行，避免漏判中途成功
    obs, rew, term, trunc, infos = env.chunk_step(chunk)
    if isinstance(obs,(list,tuple)): obs = obs[-1]
    if isinstance(obs,(list,tuple)): obs = obs[-1]
    s = infos[-1].get("success") if isinstance(infos,(list,tuple)) else None
    succ = bool(torch.as_tensor(s).any()) if s is not None else None
    print(f"GT step{c} rew={float(torch.as_tensor(rew).float().sum()):.3f} success={succ} state={[round(x,2) for x in obs['states'][0][:7].tolist()]}")
    if succ: ok=True; break
print("GT_REPLAY_SUCCESS =", ok)
