"""单环境录制指定 seed 的完整 episode（用于成功案例视频）。"""
import os, sys, json, numpy as np, torch
B="/mnt/public/zhangyuzhuo"; R=f"{B}/work/RLinf_v2"
os.environ.update({"PYTHONPATH":R,"EMBODIED_PATH":f"{R}/examples/embodiment","REPO_PATH":R,
 "LINGBOT_VLA_V2_PATH":f"{B}/venvs/rlinf_v2/lingbot-vla-v2",
 "LINGBOT_VLA_V2_MODEL_ROOT":f"{B}/models/lingbot-vla-v2-6b-robotwin",
 "LINGBOT_VLA_V2_CHECKPOINT":f"{B}/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt",
 "QWEN3VL_PATH":f"{B}/models/Qwen3-VL-4B-Instruct","ROBOTWIN_PATH":f"{B}/robotwin/RoboTwin",
 "ROBOTWIN_ASSETS_PATH":f"{B}/robotwin/RoboTwin","MUJOCO_GL":"egl","PYOPENGL_PLATFORM":"egl"})
sys.path.insert(0,R); sys.path.insert(0,f"{B}/robotwin/RoboTwin"); os.chdir(f"{R}/evaluations")
torch.cuda.set_per_process_memory_fraction(0.35,0)
from hydra import initialize, compose
import imageio.v3 as iio
TASK=os.environ.get("TASK","adjust_bottle"); SEED=int(os.environ.get("SEED","100100082"))
CFG=f"robotwin_{TASK}_lingbotvlav2_eval"; OUT=os.environ.get("OUT",f"{B}/video_frames/{TASK}_{SEED}")
os.makedirs(OUT,exist_ok=True)
json.dump({TASK:{"task_name":TASK,"success_seeds":[SEED]}},open("/tmp/_one.json","w"))
with initialize(config_path="robotwin", version_base=None):
    cfg=compose(config_name=CFG, overrides=[
      "env.eval.use_fixed_reset_state_ids=true","env.eval.seeds_path=/tmp/_one.json",
      "rollout.model.add_value_head=false","rollout.model.value_head.activation=gelu"])
from rlinf.envs import get_env_cls
from rlinf.models import get_model
CK=os.environ.get("CKPT","")
env=get_env_cls(cfg.env.eval.env_type,cfg.env.eval)(cfg=cfg.env.eval,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
model=get_model(cfg.rollout.model)
if CK and CK!="base":
    sd=torch.load(f"{CK}/actor/model_state_dict/full_weights.pt",map_location="cpu")
    model.load_state_dict({k:v for k,v in sd.items() if not k.startswith("value_head")},strict=False)
def save(o,k):
    a=np.asarray(o["main_images"][0])
    if a.ndim==3 and a.shape[0]==3: a=a.transpose(1,2,0)
    if a.max()<=1.5: a=a*255.0
    iio.imwrite(f"{OUT}/{k:04d}.png",a.astype(np.uint8))
obs,_=env.reset(); save(obs,0); k=0; done=False
for c in range(8):
    if done: break
    a,_=model.predict_action_batch({"main_images":obs["main_images"],"wrist_images":obs.get("wrist_images"),
        "states":obs["states"],"task_descriptions":obs["task_descriptions"]},mode="eval")
    acts=np.asarray(a).reshape(50,-1)[:,:14]
    for t in range(50):
        obs,_,_,_,inf=env.chunk_step(torch.as_tensor(acts[t]).unsqueeze(0).unsqueeze(0))
        if isinstance(obs,(list,tuple)): obs=obs[-1]
        k+=1; save(obs,k)
        t_=env.venv.envs[0].task
        if bool(getattr(t_,"eval_success",False)) or bool(getattr(t_,"stage_success_tag",False)):
            done=True; print(f"[成功] 第 {k} 步",flush=True); break
print(f"[结束] 帧数={k+1} 成功={done}")
