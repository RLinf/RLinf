"""零动作 / 随机动作基线：判断 21.9% 是不是"场景白送"。"""
import os, sys, json, numpy as np, torch
B="/mnt/public/zhangyuzhuo"; R=f"{B}/work/RLinf_v2"
os.environ.update({"PYTHONPATH":R,"EMBODIED_PATH":f"{R}/examples/embodiment","REPO_PATH":R,
 "LINGBOT_VLA_V2_PATH":f"{B}/venvs/rlinf_v2/lingbot-vla-v2",
 "LINGBOT_VLA_V2_MODEL_ROOT":f"{B}/models/lingbot-vla-v2-6b-robotwin",
 "LINGBOT_VLA_V2_CHECKPOINT":f"{B}/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt",
 "QWEN3VL_PATH":f"{B}/models/Qwen3-VL-4B-Instruct","ROBOTWIN_PATH":f"{B}/robotwin/RoboTwin",
 "ROBOTWIN_ASSETS_PATH":f"{B}/robotwin/RoboTwin","MUJOCO_GL":"egl","PYOPENGL_PLATFORM":"egl"})
sys.path.insert(0,R); sys.path.insert(0,f"{B}/robotwin/RoboTwin"); os.chdir(f"{R}/evaluations")
torch.cuda.set_per_process_memory_fraction(0.1,0)
from hydra import initialize, compose
from rlinf.envs import get_env_cls
TASK=os.environ.get("TASK","place_shoe")
MODE=os.environ.get("MODE","zero")           # zero | random
SEEDS=[int(x) for x in os.environ.get("SEEDS","").split(",") if x]
STEPS=int(os.environ.get("STEPS","400")); CHUNK=int(os.environ.get("CHUNK","50"))
# 随机动作的统计来自训练集
stats_path=f"{B}/datasets/robotwin2_lerobot/{'place_shoe_clean_50' if TASK=='place_shoe' else 'adjust_bottle_clean_50'}/meta/stats.json"
mu=sd=None
if MODE=="random":
    import json as _j
    st=_j.load(open(stats_path))["action"]
    mu=np.array(st["mean"],dtype=np.float32); sd=np.array(st["std"],dtype=np.float32)
def succ(infos):
    f=lambda it: bool(torch.as_tensor(it.get("success")).any()) if isinstance(it,dict) and it.get("success") is not None else False
    if isinstance(infos,dict): return f(infos)
    if isinstance(infos,(list,tuple)): return any(f(x) for x in infos)
    return False
res={}
for seed in SEEDS:
    json.dump({TASK:{"task_name":TASK,"success_seeds":[seed]}},open("/tmp/_b.json","w"))
    with initialize(config_path="robotwin", version_base=None):
        cfg=compose(config_name=f"robotwin_{TASK}_lingbotvlav2_eval", overrides=[
          "env.eval.use_fixed_reset_state_ids=true","env.eval.seeds_path=/tmp/_b.json",
          "rollout.model.add_value_head=false","rollout.model.value_head.activation=gelu"])
    env=get_env_cls(cfg.env.eval.env_type,cfg.env.eval)(cfg=cfg.env.eval,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
    env.reset(); ok=False; n=0
    while n<STEPS:
        if MODE=="zero":
            acts=np.zeros((1,CHUNK,14),dtype=np.float32)
        else:
            acts=np.clip(np.random.randn(1,CHUNK,14)*sd+mu, -3, 3).astype(np.float32)
        obs,rew,_,_,infos=env.chunk_step(torch.as_tensor(acts))
        if isinstance(obs,(list,tuple)): obs=obs[-1]
        n+=CHUNK
        try:
            t=env.venv.envs[0].task
            if bool(getattr(t,"eval_success",False)) or bool(getattr(t,"stage_success_tag",False)): ok=True; break
        except Exception: pass
        if succ(infos): ok=True; break
    res[seed]=ok
    print(f"[{MODE}] seed={seed} success={ok} steps={n}",flush=True)
print(f"[{MODE}] 汇总 {sum(res.values())}/{len(res)} = {sum(res.values())/max(1,len(res)):.3f}")
