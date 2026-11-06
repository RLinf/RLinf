"""B1/B2 混合回放：判定失败是"闭环累积"还是"单段精度不足"。
B1 = 第 0 段用模型，其余用真值；B2 = 第 0 段用真值，其余用模型。"""
import os, sys, numpy as np, torch
B="/mnt/public/zhangyuzhuo"; R=f"{B}/work/RLinf_v2"
os.environ.update({"PYTHONPATH":R,"EMBODIED_PATH":f"{R}/examples/embodiment","REPO_PATH":R,
 "LINGBOT_VLA_V2_PATH":f"{B}/venvs/rlinf_v2/lingbot-vla-v2",
 "LINGBOT_VLA_V2_MODEL_ROOT":f"{B}/models/lingbot-vla-v2-6b-robotwin",
 "LINGBOT_VLA_V2_CHECKPOINT":f"{B}/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt",
 "QWEN3VL_PATH":f"{B}/models/Qwen3-VL-4B-Instruct","ROBOTWIN_PATH":f"{B}/robotwin/RoboTwin",
 "ROBOTWIN_ASSETS_PATH":f"{B}/robotwin/RoboTwin","MUJOCO_GL":"egl","PYOPENGL_PLATFORM":"egl"})
sys.path.insert(0,R); sys.path.insert(0,f"{B}/robotwin/RoboTwin"); os.chdir(f"{R}/evaluations")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MAX_GPU_FRAC","0.25")),0)

from hydra import initialize, compose
with initialize(config_path="robotwin", version_base=None):
    cfg=compose(config_name="robotwin_click_bell_lingbotvlav2_eval", overrides=[
      "env.eval.use_fixed_reset_state_ids=true","env.eval.seeds_path=/tmp/seeds_cb0.json","env.eval.seed=0",
      "rollout.model.add_value_head=false","rollout.model.value_head.activation=gelu"])
from rlinf.envs import get_env_cls
from rlinf.models import get_model
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MODE=os.environ.get("HYBRID_MODE","B1")   # B1: 第0段模型；B2: 第0段真值
CKPT=os.environ.get("CKPT","")
env=get_env_cls(cfg.env.eval.env_type,cfg.env.eval)(cfg=cfg.env.eval,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
model=get_model(cfg.rollout.model)
if CKPT and CKPT!="base":
    sd=torch.load(f"{CKPT}/actor/model_state_dict/full_weights.pt",map_location="cpu")
    model.load_state_dict({k:v for k,v in sd.items() if not k.startswith("value_head")},strict=False)
    print("[权重] SFT:",CKPT)
else:
    print("[权重] base")
ds=LeRobotDataset(repo_id="click_bell_clean_50",root=f"{B}/datasets/robotwin2_lerobot/click_bell_clean_50")
gt=np.stack([ds[i]["action"].numpy() for i in range(400)])   # (400,14)
obs,_=env.reset()
ANY=False
for c in range(8):
    use_model = (c==0) if MODE=="B1" else (c>0)
    if use_model:
        a,_=model.predict_action_batch({"main_images":obs["main_images"],"wrist_images":obs.get("wrist_images"),
            "states":obs["states"],"task_descriptions":obs["task_descriptions"]},mode="eval")
        acts=torch.as_tensor(np.asarray(a)).reshape(1,50,-1)
        src="模型"
    else:
        acts=torch.as_tensor(gt[c*50:(c+1)*50]).unsqueeze(0)
        src="真值"
    obs,rew,done,trunc,infos=env.chunk_step(acts)
    if isinstance(obs,(list,tuple)): obs=obs[-1]
    flags=[]
    if isinstance(infos,(list,tuple)):
        for it in infos:
            if isinstance(it,dict) and it.get("success") is not None:
                flags.append(bool(torch.as_tensor(it["success"]).any()))
    succ=any(flags) if flags else False
    ANY=ANY or succ
    print(f"[{MODE}] chunk{c} 动作={src} success={succ} 累计成功={ANY}",flush=True)
print(f"[{MODE}] 最终结果 = {ANY}")
