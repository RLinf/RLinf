"""打印接触窗口附近每一步的 info 内容与判定要素，确认 success 为何不触发。"""
import os, sys, numpy as np, torch
B="/mnt/public/zhangyuzhuo"; R=f"{B}/work/RLinf_v2"
os.environ.update({"PYTHONPATH":R,"EMBODIED_PATH":f"{R}/examples/embodiment","REPO_PATH":R,
 "LINGBOT_VLA_V2_PATH":f"{B}/venvs/rlinf_v2/lingbot-vla-v2",
 "LINGBOT_VLA_V2_MODEL_ROOT":f"{B}/models/lingbot-vla-v2-6b-robotwin",
 "LINGBOT_VLA_V2_CHECKPOINT":f"{B}/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt",
 "QWEN3VL_PATH":f"{B}/models/Qwen3-VL-4B-Instruct","ROBOTWIN_PATH":f"{B}/robotwin/RoboTwin",
 "ROBOTWIN_ASSETS_PATH":f"{B}/robotwin/RoboTwin","MUJOCO_GL":"egl","PYOPENGL_PLATFORM":"egl"})
sys.path.insert(0,R); sys.path.insert(0,f"{B}/robotwin/RoboTwin"); os.chdir(f"{R}/evaluations")
torch.cuda.set_per_process_memory_fraction(0.05,0)
from hydra import initialize, compose
with initialize(config_path="robotwin", version_base=None):
    cfg=compose(config_name="robotwin_click_bell_lingbotvlav2_eval", overrides=[
      "env.eval.use_fixed_reset_state_ids=true","env.eval.seeds_path=/tmp/seeds_cb0.json","env.eval.seed=0"])
from rlinf.envs import get_env_cls
from lerobot.datasets.lerobot_dataset import LeRobotDataset
env=get_env_cls(cfg.env.eval.env_type,cfg.env.eval)(cfg=cfg.env.eval,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
obs,_=env.reset(); task=env.venv.envs[0].task
ds=LeRobotDataset(repo_id="click_bell_clean_50",root=f"{B}/datasets/robotwin2_lerobot/click_bell_clean_50")
acts=np.stack([ds[i]["action"].numpy() for i in range(81)])
for k in range(81):
    obs,rew,_,_,inf=env.chunk_step(torch.as_tensor(acts[k]).unsqueeze(0).unsqueeze(0))
    if isinstance(obs,(list,tuple)): obs=obs[-1]
    if 55<=k<=70:
        d=inf if isinstance(inf,dict) else (inf[0] if isinstance(inf,(list,tuple)) else {})
        print(f"step{k}: keys={list(d.keys())[:6]} rew={float(torch.as_tensor(rew).float().sum().item()):.3f} "
              f"gripper_close={task.is_left_gripper_close()} stage={getattr(task,'stage_success_tag',None)}")
