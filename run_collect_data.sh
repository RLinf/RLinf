ray stop --force >/dev/null 2>&1 || true; pkill -TERM -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true; sleep 2; pkill -KILL -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true
export RLINF_TASK_DESCRIPTION="replay test"
bash examples/embodiment/collect_data.sh \
    realworld_collect_data_ros2_gello_dual_franka_pnp.yaml \
    runner.num_data_episodes=1
