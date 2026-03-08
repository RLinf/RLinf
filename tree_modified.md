# Pi06 PR File Tree (vs upstream 909494e)

61 new files, 10 modified files

```
├── .gitignore  [MOD]
├── examples/
│   ├── embodiment/
│   │   ├── config/
│   │   │   ├── one_iter_debug_libero10.yaml  [NEW]
│   │   │   └── one_iter_debug_libero10_test.yaml  [NEW]
│   │   ├── eval_embodiment.sh  [MOD]
│   │   ├── run_debug_one_iter.sh  [NEW]
│   │   ├── run_embodiment.sh  [MOD]
│   │   └── train_debug_one_iter.py  [NEW]
│   ├── process/
│   │   ├── __init__.py  [NEW]
│   │   ├── compute_advantages.py  [NEW]
│   │   ├── compute_returns.py  [NEW]
│   │   ├── config/
│   │   │   ├── compute_advantages.yaml  [NEW]
│   │   │   ├── compute_advantages_test.yaml  [NEW]
│   │   │   ├── compute_returns.yaml  [NEW]
│   │   │   └── compute_returns_test.yaml  [NEW]
│   │   ├── recompute_advantages_from_value_reward.py  [NEW]
│   │   ├── run_compute_advantages.sh  [NEW]
│   │   └── run_compute_returns.sh  [NEW]
│   └── sft/
│       ├── config/
│       │   ├── libero_cfg_openpi.yaml  [NEW]
│       │   ├── libero_cfg_openpi_test.yaml  [NEW]
│       │   ├── libero_sft_value.yaml  [NEW]
│       │   ├── libero_sft_value_test.yaml  [NEW]
│       │   └── model/
│       │       ├── pi0_5.yaml  [NEW]
│       │       └── vla_lib_value_model.yaml  [NEW]
│       ├── run_cfg_sft.sh  [NEW]
│       ├── run_vla_lib_sft.sh  [NEW]
│       ├── train_cfg_sft.py  [NEW]
│       └── train_vla_lib_sft.py  [NEW]
├── install_for_use_vla_lib.sh  [NEW]
├── rlinf/
│   ├── config.py  [MOD]
│   ├── data/
│   │   ├── datasets/
│   │   │   └── utils.py  [NEW]
│   │   ├── lerobot_writer.py  [MOD]
│   │   └── rollout_data_collector.py  [NEW]
│   ├── datasets/
│   │   ├── __init__.py  [NEW]
│   │   ├── config.py  [NEW]
│   │   ├── lerobot/
│   │   │   ├── __init__.py  [NEW]
│   │   │   ├── config.py  [NEW]
│   │   │   ├── lerobot_dataset.py  [NEW]
│   │   │   ├── libero.py  [NEW]
│   │   │   ├── normalize.py  [NEW]
│   │   │   └── transforms.py  [NEW]
│   │   ├── mixture_datasets.py  [NEW]
│   │   ├── rl_dataset.py  [NEW]
│   │   ├── value_dataset.py  [NEW]
│   │   └── value_transforms.py  [NEW]
│   ├── envs/
│   │   └── libero/
│   │       └── libero_env.py  [MOD]
│   ├── models/
│   │   ├── __init__.py  [MOD]
│   │   └── embodiment/
│   │       ├── openpi/
│   │       │   └── dataconfig/
│   │       │       └── __init__.py  [MOD]
│   │       ├── openpi_cfg/
│   │       │   ├── __init__.py  [NEW]
│   │       │   └── openpi_cfg_action_model.py  [NEW]
│   │       └── vla_lib_value_model/
│   │           ├── __init__.py  [NEW]
│   │           ├── base_policy.py  [NEW]
│   │           ├── configs.py  [NEW]
│   │           ├── configuration.py  [NEW]
│   │           ├── data_collator.py  [NEW]
│   │           ├── modeling_pi05_critic.py  [NEW]
│   │           ├── paligemma_with_multi_expert.py  [NEW]
│   │           ├── processing.py  [NEW]
│   │           ├── value_policy.py  [NEW]
│   │           └── value_policy_config.py  [NEW]
│   ├── runners/
│   │   └── debug_pi06_runner.py  [NEW]
│   ├── utils/
│   │   └── ckpt_convertor/
│   │       └── fsdp_convertor/
│   │           └── config/
│   │               └── fsdp_vla_lib_model_convertor.yaml  [NEW]
│   └── workers/
│       ├── actor/
│       │   └── debug_fsdp_actor_worker_cfg.py  [NEW]
│       ├── cfg/
│       │   ├── __init__.py  [NEW]
│       │   ├── fsdp_cfg_worker.py  [NEW]
│       │   └── utils.py  [NEW]
│       ├── env/
│       │   └── env_worker.py  [MOD]
│       ├── rollout/
│       │   └── hf/
│       │       └── huggingface_worker.py  [MOD]
│       └── vla_lib_sft/
│           ├── __init__.py  [NEW]
│           └── fsdp_value_sft_worker.py  [NEW]
├── tree_modified.md  [NEW]
└── use_guidance.md  [NEW]
```
