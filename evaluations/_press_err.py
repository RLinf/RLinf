"""按压段姿态误差：GT 执行第 0 段后，让模型预测第 1 段，比较停住姿态与示范按压姿态。"""
import os, sys, numpy as np, torch
B="/mnt/public/zhangyuzhuo"; R=f"{B}/work/RLinf_v2"
os.environ.update({"PYTHONPATH":R,"EMBODIED_PATH":f"{R}/examples/embodiment","REPO_PATH":R,
 "LINGBOT_VLA_V2_PATH":f"{B}/venvs/rlinf_v2/lingbot-vla-v2",
 "LINGBOT_VLA_V2_MODEL_ROOT":f"{B}/models/lingbot-vla-v2-6b-robotwin",
 "LINGBOT_VLA_V2_CHECKPOINT":f"{B}/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt",
 "QWEN3VL_PATH":f"{B}/models/Qwen3-VL-4B-Instruct","ROBOTWIN_PATH":f"{B}/robotwin/RoboTwin",
 "ROBOTWIN_ASSETS_PATH":f"{B}/robotwin/RoboTwin","MUJOCO_GL":"egl","PYOPENGL_PLATFORM":"egl"})
sys.path.insert(0,R); sys.path.insert(0,f"{B}/robotwin/RoboTwin"); os.chdir(f"{R}/evaluations")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MAX_GPU_FRAC","0.35")),0)
from hydra import initialize, compose
with initialize(config_path="robotwin", version_base=None):
    cfg=compose(config_name="robotwin_click_bell_lingbotvlav2_eval", overrides=[
      "env.eval.use_fixed_reset_state_ids=true","env.eval.seeds_path=/tmp/seeds_cb0.json","env.eval.seed=0",
      "rollout.model.add_value_head=false","rollout.model.value_head.activation=gelu"])
from rlinf.envs import get_env_cls
from rlinf.models import get_model
from lerobot.datasets.lerobot_dataset import LeRobotDataset
CK=os.environ.get("CKPT","")
env=get_env_cls(cfg.env.eval.env_type,cfg.env.eval)(cfg=cfg.env.eval,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
model=get_model(cfg.rollout.model)
if CK and CK!="base":
    sd=torch.load(f"{CK}/actor/model_state_dict/full_weights.pt",map_location="cpu")
    model.load_state_dict({k:v for k,v in sd.items() if not k.startswith("value_head")},strict=False)
ds=LeRobotDataset(repo_id="click_bell_clean_50",root=f"{B}/datasets/robotwin2_lerobot/click_bell_clean_50")
gt=np.stack([ds[i]["action"].numpy() for i in range(81)])
obs,_=env.reset()
obs,_,_,_,_=env.chunk_step(torch.as_tensor(gt[0:50]).unsqueeze(0))
if isinstance(obs,(list,tuple)): obs=obs[-1]
a,_=model.predict_action_batch({"main_images":obs["main_images"],"wrist_images":obs.get("wrist_images"),
    "states":obs["states"],"task_descriptions":obs["task_descriptions"]},mode="eval")
pred=np.asarray(a).reshape(50,-1)[:,:14]
hold_pred=pred[-10:].mean(0)          # 模型停住的姿态
hold_gt=gt[70:80,:14].mean(0)         # 示范按压保持姿态
err=np.abs(hold_pred-hold_gt)
print(f"[按压误差] 平均 {err[:13].mean():.4f} rad | 各关节 {np.round(err[:7],3).tolist()}")
print(f"[按压误差] 模型 {np.round(hold_pred[:7],3).tolist()}")
print(f"[按压误差] 真值 {np.round(hold_gt[:7],3).tolist()}")
