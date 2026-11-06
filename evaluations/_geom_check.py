"""几何一致性检查：(1) 环境末端位姿 vs 数据 endpose（验底座/运动学）
(2) 继续下压时铃铛是否移动（验夹爪碰撞体是否挡住）"""
import os, sys, numpy as np, torch, h5py
B="/mnt/public/zhangyuzhuo"; R=f"{B}/work/RLinf_v2"
os.environ.update({"PYTHONPATH":R,"EMBODIED_PATH":f"{R}/examples/embodiment","REPO_PATH":R,
 "LINGBOT_VLA_V2_PATH":f"{B}/venvs/rlinf_v2/lingbot-vla-v2",
 "LINGBOT_VLA_V2_MODEL_ROOT":f"{B}/models/lingbot-vla-v2-6b-robotwin",
 "LINGBOT_VLA_V2_CHECKPOINT":f"{B}/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt",
 "QWEN3VL_PATH":f"{B}/models/Qwen3-VL-4B-Instruct","ROBOTWIN_PATH":f"{B}/robotwin/RoboTwin",
 "ROBOTWIN_ASSETS_PATH":f"{B}/robotwin/RoboTwin","MUJOCO_GL":"egl","PYOPENGL_PLATFORM":"egl"})
sys.path.insert(0,R); sys.path.insert(0,f"{B}/robotwin/RoboTwin"); os.chdir(f"{R}/evaluations")
torch.cuda.set_per_process_memory_fraction(0.3,0)
from hydra import initialize, compose
with initialize(config_path="robotwin", version_base=None):
    cfg=compose(config_name="robotwin_click_bell_lingbotvlav2_eval", overrides=[
      "env.eval.use_fixed_reset_state_ids=true","env.eval.seeds_path=/tmp/seeds_cb0.json","env.eval.seed=0"])
from rlinf.envs import get_env_cls
from lerobot.datasets.lerobot_dataset import LeRobotDataset
env=get_env_cls(cfg.env.eval.env_type,cfg.env.eval)(cfg=cfg.env.eval,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
obs,_=env.reset()
task=env.venv.envs[0].task
h=h5py.File(f"{B}/datasets/robotwin_raw/click_bell/aloha-agilex_clean_50/data/episode0.hdf5","r")
ep=np.array(h["endpose/left_endpose"])          # (81,7) xyz+quat?
print("[数据] endpose 末帧 xyz =", np.round(ep[-1,:3],4).tolist())
print("[数据] endpose z 范围 =", round(float(ep[:,2].min()),4), "~", round(float(ep[:,2].max()),4))
ds=LeRobotDataset(repo_id="click_bell_clean_50",root=f"{B}/datasets/robotwin2_lerobot/click_bell_clean_50")
acts=np.stack([ds[i]["action"].numpy() for i in range(81)])
ez=[]
for k in range(81):
    obs,_,_,_,_=env.chunk_step(torch.as_tensor(acts[k]).unsqueeze(0).unsqueeze(0))
    if isinstance(obs,(list,tuple)): obs=obs[-1]
    try: ez.append(task.robot.get_left_ee_pose()[:3])
    except Exception as e: ez.append(np.zeros(3)); 
ez=np.array(ez)
print("[环境] EE 末帧 xyz =", np.round(ez[-1],4).tolist())
print("[对比] 末端轨迹平均差 =", round(float(np.abs(ez-ep[:,:3]).mean()),4),
      " 最大差 =", round(float(np.abs(ez-ep[:,:3]).max()),4), " (米)")
print("[对比] z 方向平均差 =", round(float((ez[:,2]-ep[:,2]).mean()),4))
# 继续下压，看铃铛是否响应
bp0=np.array(task.bell.get_pose().p)
print("[铃铛] 下压前 z =", round(float(bp0[2]),4))
last=acts[-1].copy()
for i in range(15):
    a=last.copy(); a[1]+=0.01; a[2]-=0.01
    obs,rew,_,_,inf=env.chunk_step(torch.as_tensor(a).unsqueeze(0).unsqueeze(0))
    if isinstance(obs,(list,tuple)): obs=obs[-1]
    bp=np.array(task.bell.get_pose().p)
    s=False
    if isinstance(inf,dict): s=bool(torch.as_tensor(inf.get("success",False)).any())
    print(f"[下压{i+1}] 铃铛z={bp[2]:.4f} (Δ{bp[2]-bp0[2]:+.4f}) success={s}")
    last=a
