"""逐步执行并保存每一帧，用于制作对比视频。MODE=gt 或 model。"""
import os, sys, numpy as np, torch
B="/mnt/public/zhangyuzhuo"; R=f"{B}/work/RLinf_v2"
os.environ.update({"PYTHONPATH":R,"EMBODIED_PATH":f"{R}/examples/embodiment","REPO_PATH":R,
 "LINGBOT_VLA_V2_PATH":f"{B}/venvs/rlinf_v2/lingbot-vla-v2",
 "LINGBOT_VLA_V2_MODEL_ROOT":f"{B}/models/lingbot-vla-v2-6b-robotwin",
 "LINGBOT_VLA_V2_CHECKPOINT":f"{B}/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt",
 "QWEN3VL_PATH":f"{B}/models/Qwen3-VL-4B-Instruct","ROBOTWIN_PATH":f"{B}/robotwin/RoboTwin",
 "ROBOTWIN_ASSETS_PATH":f"{B}/robotwin/RoboTwin","MUJOCO_GL":"egl","PYOPENGL_PLATFORM":"egl"})
sys.path.insert(0,R); sys.path.insert(0,f"{B}/robotwin/RoboTwin"); os.chdir(f"{R}/evaluations")
torch.cuda.set_per_process_memory_fraction(0.4,0)
from hydra import initialize, compose
with initialize(config_path="robotwin", version_base=None):
    cfg=compose(config_name="robotwin_click_bell_lingbotvlav2_eval", overrides=[
      "env.eval.use_fixed_reset_state_ids=true","env.eval.seeds_path=/tmp/seeds_cb0.json","env.eval.seed=0",
      "rollout.model.add_value_head=false","rollout.model.value_head.activation=gelu"])
from rlinf.envs import get_env_cls
from rlinf.models import get_model
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import imageio.v3 as iio
MODE=os.environ.get("MODE","model"); OUT=os.environ.get("OUT",f"{B}/video_frames/{MODE}")
os.makedirs(OUT,exist_ok=True)
CK=os.environ.get("CKPT","")
env=get_env_cls(cfg.env.eval.env_type,cfg.env.eval)(cfg=cfg.env.eval,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
model=None
if MODE=="model":
    model=get_model(cfg.rollout.model)
    if CK and CK!="base":
        sd=torch.load(f"{CK}/actor/model_state_dict/full_weights.pt",map_location="cpu")
        model.load_state_dict({k:v for k,v in sd.items() if not k.startswith("value_head")},strict=False)
def save(o,k):
    a=np.asarray(o["main_images"][0])
    if a.ndim==3 and a.shape[0]==3: a=a.transpose(1,2,0)
    if a.max()<=1.5: a=a*255.0
    iio.imwrite(f"{OUT}/{k:04d}.png",a.astype(np.uint8))
obs,_=env.reset(); save(obs,0)
if MODE=="gt":
    ds=LeRobotDataset(repo_id="click_bell_clean_50",root=f"{B}/datasets/robotwin2_lerobot/click_bell_clean_50")
    acts=np.stack([ds[i]["action"].numpy() for i in range(81)])
    for k in range(81):
        obs,_,_,_,_=env.chunk_step(torch.as_tensor(acts[k]).unsqueeze(0).unsqueeze(0))
        if isinstance(obs,(list,tuple)): obs=obs[-1]
        save(obs,k+1)
else:
    k=0
    for c in range(8):
        a,_=model.predict_action_batch({"main_images":obs["main_images"],"wrist_images":obs.get("wrist_images"),
            "states":obs["states"],"task_descriptions":obs["task_descriptions"]},mode="eval")
        acts=np.asarray(a).reshape(50,-1)[:,:14]
        for t in range(50):
            obs,_,_,_,_=env.chunk_step(torch.as_tensor(acts[t]).unsqueeze(0).unsqueeze(0))
            if isinstance(obs,(list,tuple)): obs=obs[-1]
            k+=1; save(obs,k)
print(f"[record] {MODE} 帧数 = {k+1}")
