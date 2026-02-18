# 04 - verl Training Config Blueprint

## 1. Khung config multi-turn tool-calling

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  train_files: /data/toolcall/train.parquet
  val_files: /data/toolcall/val.parquet
  return_raw_chat: true
  max_prompt_length: 2048
  max_response_length: 1024
  train_batch_size: 128
  filter_overlong_prompts: true

algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false
  norm_adv_by_std_in_grpo: true

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-3B-Instruct
    use_remove_padding: true
    enable_gradient_checkpointing: true
  actor:
    strategy: fsdp2
    ppo_mini_batch_size: 128
    ppo_micro_batch_size_per_gpu: 8
    use_kl_loss: true
    kl_loss_coef: 0.001
  rollout:
    name: sglang
    n: 8
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.8
    multi_turn:
      enable: true
      tool_config_path: /data/tool_config.yaml
      max_assistant_turns: 6

trainer:
  logger: [console, wandb]
  project_name: verl-toolcall-agent
  experiment_name: grpo-mt-v1
  n_gpus_per_node: 8
  nnodes: 1
  test_freq: 20
  save_freq: 50
  total_epochs: 5
```

## 2. Các điểm nhạy nhất

1. `data.return_raw_chat=True`.
2. `rollout.multi_turn.enable=True` và `tool_config_path` đúng.
3. `rollout.n` tăng thì token budget tăng bậc cao.
4. `ppo_micro_batch_size_per_gpu` là núm chống OOM đầu tiên.

## 3. Done Criteria

- Config pass sanity check.
- Chạy smoke test vài steps không lỗi parse/tool.
- Có baseline metrics đầu tiên để so sánh.
