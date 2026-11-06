"""环境审计：底座/桌面/铃铛 z、夹爪是否被碰撞体挡住、腕部相机输入。"""
import os, sys, numpy as np, torch
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
import imageio.v3 as iio
env=get_env_cls(cfg.env.eval.env_type,cfg.env.eval)(cfg=cfg.env.eval,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
obs,_=env.reset()
# 找到底层 task
task=None
for path in ["venv","env","_env"]:
    v=getattr(env,path,None)
    if v is None: continue
    for sub in ["envs","_envs"]:
        lst=getattr(v,sub,None)
        if lst:
            e0=lst[0]
            task=getattr(e0,"task",None) or getattr(getattr(e0,"unwrapped",None),"task",None)
            if task: break
    if task: break
print("task:",type(task).__name__ if task else "未找到")
if task:
    try: print("[底座] robot base pose z =", round(float(task.robot.get_pose().p[2]),4))
    except Exception as e: print("[底座] 读取失败:",e)
    try: print("[桌面] table z =", round(float(task.table_z),4) if hasattr(task,"table_z") else "n/a")
    except Exception as e: print("[桌面]",e)
    try:
        bp=task.bell.get_pose().p
        print("[铃铛] pose =",[round(float(x),4) for x in bp])
        print("[铃铛] 接触点0 =",[round(float(x),4) for x in task.bell.get_contact_point(0)[:3]])
    except Exception as e: print("[铃铛]",e)
# 腕部相机对比
def hwc(t):
    a=t.detach().cpu().numpy() if hasattr(t,"detach") else np.asarray(t)
    if a.ndim==4: a=a[0]
    if a.ndim==3 and a.shape[0]==3: a=a.transpose(1,2,0)
    if a.max()<=1.5: a=a*255.0
    return a.astype(np.uint8)
w=obs.get("wrist_images")
print("[腕部] wrist_images shape:", np.asarray(w).shape)
if w is not None:
    W=hwc(w[0][0] if np.asarray(w).ndim==5 else w[0])
    iio.imwrite("/tmp/env_wrist0.png",W)
    ds=LeRobotDataset(repo_id="click_bell_clean_50",root=f"{B}/datasets/robotwin2_lerobot/click_bell_clean_50")
    D=hwc(ds[0]["observation.images.cam_left_wrist"])
    iio.imwrite("/tmp/ds_wrist0.png",D)
    from PIL import Image
    a=Image.open("/tmp/env_wrist0.png").convert("RGB"); b=Image.open("/tmp/ds_wrist0.png").convert("RGB").resize(a.size)
    d=np.abs(np.asarray(a,float)-np.asarray(b,float)).mean()
    print(f"[腕部] 与数据首帧平均差 = {d:.2f}/255")
