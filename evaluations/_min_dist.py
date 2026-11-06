"""沿真值轨迹逐步执行，记录"接触点 vs 铃铛顶部中心"的最小距离（对应 success_once 判定）。"""
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
best=(9.9,9.9,-1); succ_at=None
for k in range(81):
    obs,rew,_,_,inf=env.chunk_step(torch.as_tensor(acts[k]).unsqueeze(0).unsqueeze(0))
    if isinstance(obs,(list,tuple)): obs=obs[-1]
    try:
        b=task.bell.get_contact_point(0)[:3]
        cps=task.get_gripper_actor_contact_position("050_bell")
        for cp in cps:
            dz=abs(float(cp[2]-b[2])); dxy=float(np.linalg.norm([cp[0]-b[0],cp[1]-b[1]]))
            if dxy<best[1] or (abs(dxy-best[1])<1e-6 and dz<best[0]):
                best=(min(best[0],dz),min(best[1],dxy),k)
    except Exception: pass
    s=False
    if isinstance(inf,dict): s=bool(torch.as_tensor(inf.get("success",False)).any())
    if s and succ_at is None: succ_at=k
print(f"[GT轨迹] 最小 dz = {best[0]*100:.2f} cm, 最小 dxy = {best[1]*100:.2f} cm (第 {best[2]} 步)  窗口 3.0/2.5 cm")
print(f"[GT轨迹] success 首次触发于第 {succ_at} 步" if succ_at is not None else "[GT轨迹] 全程未触发 success")
