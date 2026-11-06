"""查两件事：(1) 底座/运动学零点 (2) 夹爪碰撞体是否挡住下压"""
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
obs,_=env.reset()
task=env.venv.envs[0].task
r=task.robot
# --- 1) 底座 ---
for attr in ["root_pose","get_root_pose","get_pose","entity"]:
    if hasattr(r,attr):
        try:
            v=getattr(r,attr)() if callable(getattr(r,attr)) else getattr(r,attr)
            print(f"[底座] {attr} =", v if not hasattr(v,"p") else [round(float(x),4) for x in v.p])
        except Exception as e: print(f"[底座] {attr} 失败 {e}")
print("[底座] 左臂初始关节 =", np.round(np.asarray(task.robot.get_left_arm_jointState() if hasattr(task.robot,'get_left_arm_jointState') else []),3).tolist()[:6])
# --- 2) 夹爪碰撞体：从按压位姿继续下压，观察 EE 与接触点 ---
ds=LeRobotDataset(repo_id="click_bell_clean_50",root=f"{B}/datasets/robotwin2_lerobot/click_bell_clean_50")
acts=np.stack([ds[i]["action"].numpy() for i in range(81)])
for k in range(81):   # 先按真值走到按压
    obs,_,_,_,_=env.chunk_step(torch.as_tensor(acts[k]).unsqueeze(0).unsqueeze(0))
    if isinstance(obs,(list,tuple)): obs=obs[-1]
ee0=np.array(task.robot.get_left_ee_pose()[:3])
print("[准备] 按压位姿 EE =", np.round(ee0,4).tolist())
last=acts[-1].copy()
print("[下压] 命令额外下压，观察 EE 是否跟随、是否出现接触")
for i in range(10):
    a=last.copy(); a[1]+=0.03; a[2]-=0.03
    obs,rew,_,_,inf=env.chunk_step(torch.as_tensor(a).unsqueeze(0).unsqueeze(0))
    if isinstance(obs,(list,tuple)): obs=obs[-1]
    ee=np.array(task.robot.get_left_ee_pose()[:3])
    try: cps=task.get_gripper_actor_contact_position("050_bell"); nc=len(cps)
    except Exception: nc=-1
    s=False
    if isinstance(inf,dict): s=bool(torch.as_tensor(inf.get("success",False)).any())
    print(f"[下压{i+1}] EE_z={ee[2]:.4f} (Δ{ee[2]-ee0[2]:+.4f} m) 接触点数={nc} success={s}")
    last=a
