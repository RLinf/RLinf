# export RLINF_NODE_RANK=2
export RLINF_NODE_RANK=0
unset PYTHONPATH
source /home/ysa/dagger/RLinf/.venv/bin/activate
export PYTHONPATH="$PYTHONPATH:/home/ysa/dagger/RLinf" ##
export RLINF_COMM_NET_DEVICES=rlinf
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# ray start --head --node-ip-address=10.126.126.11