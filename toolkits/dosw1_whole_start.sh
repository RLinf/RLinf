#!/bin/bash
# start_all_arms_final.sh
# 功能：弹出两个独立终端
#   终端1：tmux 2×2 跑 4 个 airbot_server
#   终端2：conda 环境 + main.py，执行完后保持打开

SESSION="airbot_servers"
  if [ $(docker ps -q | wc -l) -gt 0 ]; then
    echo "waiting, stop all dockers...."
    docker stop $(docker ps -q)
  fi

########## 终端 1：tmux 4 分屏 + teachbag_reciever ##########
gnome-terminal -- bash -c "
  # 如果会话已存在先杀掉，避免重复
  tmux kill-session -t $SESSION 2>/dev/null || true
  tmux new-session -d -s $SESSION -n airbots
  tmux set-option -t $SESSION mouse on
  tmux set-option -t $SESSION remain-on-exit off

  tmux send-keys -t $SESSION:airbots 'conda activate airbot && sleep 1 && airbot_server -i can_left_lead -p 50050' C-m
  sleep 1
  
  tmux split-window -v -t $SESSION:airbots
  tmux send-keys -t $SESSION:airbots 'conda activate airbot && sleep 1 && airbot_server -i can_left -p 50051' C-m
  sleep 1
  
  tmux select-pane -t 0
  tmux split-window -h -t $SESSION:airbots
  tmux send-keys -t $SESSION:airbots 'conda activate airbot && sleep 1 && airbot_server -i can_right_lead -p 50052' C-m
  sleep 1
  
  tmux select-pane -t 1
  tmux split-window -h -t $SESSION:airbots
  tmux send-keys -t $SESSION:airbots 'conda activate airbot && sleep 1 && airbot_server -i can_right -p 50053' C-m
  sleep 1
  

  # 自动平铺
  tmux select-layout -t $SESSION:airbots tiled
  exec tmux attach-session -t $SESSION
"

# ########## 终端 2：独立运行 main.py ##########
# gnome-terminal -- bash -ic "
#   sleep 15
#   conda activate airbot_data
#   # cd /home/ubuntu22/hq_v1/product-demo-5.1.6.8a2/airbot-data-5.1.6.8a2/data-collection/airbot_data_collection
#   # python3 main.py --path defaults/config_setup.yaml --dataset.directory example
#   python3 test_task_follow.py -lp 50050 50052 -fp 50051 50053
#   exec bash
# "
