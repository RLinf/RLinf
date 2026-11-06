"""评测 + 记录每条轨迹的最小接触距离（对应 success_once 判定）。"""
import os, sys, numpy as np, torch
B="/mnt/public/zhangyuzhuo"; R=f"{B}/work/RLinf_v2"
os.environ.update({"PYTHONPATH":R,"EMBODIED_PATH":f"{R}/examples/embodiment","REPO_PATH":R,
 "LINGBOT_VLA_V2_PATH":f"{B}/venvs/rlinf_v2/lingbot-vla-v2",
 "LINGBOT_VLA_V2_MODEL_ROOT":f"{B}/models/lingbot-vla-v2-6b-robotwin",
 "LINGBOT_VLA_V2_CHECKPOINT":f"{B}/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt",
 "QWEN3VL_PATH":f"{B}/models/Qwen3-VL-4B-Instruct","ROBOTWIN_PATH":f"{B}/robotwin/RoboTwin",
 "ROBOTWIN_ASSETS_PATH":f"{B}/robotwin/RoboTwin","MUJOCO_GL":"egl","PYOPENGL_PLATFORM":"egl"})
sys.path.insert(0,R); sys.path.insert(0,f"{B}/robotwin/RoboTwin"); os.chdir(f"{R}/evaluations")
torch.cuda.set_per_process_memory_fraction(0.28,0)
from hydra import initialize, compose
import json
SP=os.environ.get("SEEDS_PATH","/tmp/seed_off_0.json")
with initialize(config_path="robotwin", version_base=None):
    cfg=compose(config_name="robotwin_click_bell_lingbotvlav2_eval", overrides=[
      "env.eval.use_fixed_reset_state_ids=true",f"env.eval.seeds_path={SP}",
      "rollout.model.add_value_head=false","rollout.model.value_head.activation=gelu"])
from rlinf.envs import get_env_cls
from rlinf.models import get_model
CK=os.environ.get("CKPT","")
env=get_env_cls(cfg.env.eval.env_type,cfg.env.eval)(cfg=cfg.env.eval,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
model=get_model(cfg.rollout.model)
if CK and CK!="base":
    sd=torch.load(f"{CK}/actor/model_state_dict/full_weights.pt",map_location="cpu")
    model.load_state_dict({k:v for k,v in sd.items() if not k.startswith("value_head")},strict=False)
obs,_=env.reset(); task=env.venv.envs[0].task
succ_seen=False; min_dz=9.9; min_dxy=9.9; ncontact=0
for c in range(8):
    a,_=model.predict_action_batch({"main_images":obs["main_images"],"wrist_images":obs.get("wrist_images"),
        "states":obs["states"],"task_descriptions":obs["task_descriptions"]},mode="eval")
    acts=np.asarray(a).reshape(50,-1)[:,:14]
    for t in range(50):
        obs,rew,_,_,inf=env.chunk_step(torch.as_tensor(acts[t]).unsqueeze(0).unsqueeze(0))
        if isinstance(obs,(list,tuple)): obs=obs[-1]
        if bool(getattr(task,"eval_success",False)) or bool(getattr(task,"stage_success_tag",False)):
            succ_seen=True
        try:
            b=task.bell.get_contact_point(0)[:3]
            cps=task.get_gripper_actor_contact_position("050_bell")
            if cps: ncontact+=1
            for cp in cps:
                dz=abs(float(cp[2]-b[2])); dxy=float(np.linalg.norm([cp[0]-b[0],cp[1]-b[1]]))
                if dxy<0.025: min_dz=min(min_dz,dz)
                min_dxy=min(min_dxy,dxy)
        except Exception: pass
print(f"RESULT success={succ_seen} min_dz={min_dz*100:.2f}cm min_dxy={min_dxy*100:.2f}cm 有接触步数={ncontact}")
