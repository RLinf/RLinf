# Dependencies
Refer to RLinf docs. For small models, I typically use the virtual env for openvla, but openvla-oft and openpi are also OK.

# Start cmd
Synchronized 版本
```
bash examples/embodiment/run_embodiment.sh [yaml-name]
```
Async 版本
```
bash examples/embodiment/run_async.sh [yaml-name]
```

# Some Issues
1. 目前最新版本跑VLA模型会有点问题。主要是因为在配真机环境的时候注释掉了一些 import 。如果遇到了相关问题，把相关 import 改回来即可。非常抱歉！！

# Supported Feartures
## Training 
### Sim
1. SAC
  - SAC+MLP: examples/embodiment/config/maniskill_sac_mlp.yaml
    能看到 success rate 上涨到 1
  - SAC+CNN: examples/embodiment/config/maniskill_sac_cnn.yaml
    reward 暂时上涨较慢。目前发现把 lr 改大到 1e-3 能有明显增快。最佳 cfg 未上传。
  - SAC+OpenVLA-OFT：examples/embodiment/config/maniskill_sac_openvlaoft.yaml
    会训崩，暂时不计划调。难点主要在于离散的 token 和 action chunk。
    
2. Cross-Q
  - CrossQ+MLP: examples/embodiment/config/maniskill_crossq_mlp.yaml
    shu'ang 写的，可以看到比 SAC+MLP 更快
  - CrossQ+CNN:
    code 已经支持了，yaml 还没 commit and push
    效果没有仔细和 SAC 对比

3. RLPD (i.e. SAC + demo buffer)
  - CNN: examples/embodiment/config/maniskill_rlpd_cnn.yaml
    lr 调大效果更好。已经比 SAC+CNN 好了。
  - collect sim expert data：写完了，没 commit and push

4. Async Training
  - SAC+MLP: examples/embodiment/config/maniskill_sac_mlp_async.yaml
  - SAC+CNN: 没上传
  - CrossQ+CNN：没上传

## Environment
1. Real Franka Env
  主要只支持了无人机插电的任务
  - Model(支持 inference): 
    - CNN
    - OpenPi
  - Model(支持 Train)
    - CNN
  - Algorithm
    - SAC: examples/embodiment/config/real_sac_cnn.yaml
      dense reward 能看到逐渐学会
      dense reward 目前是在环境里 hardcode 去修改的，没有从 cfg 修改
    - RLPD
      能跑通但效果不行

# TODO
1. Human-in-the-loop
  空间鼠标/键盘
2. Collect data in real
3. 其他 serl 的环境