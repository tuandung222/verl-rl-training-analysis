# Ví dụ: Qwen + PPO cho tool-calling multiturn (trên `verl`)

Ví dụ này bám vào các script/config upstream:

- `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh`
- `examples/sglang_multiturn/config/gsm8k_multiturn_grpo.yaml`
- `examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml`

## 1. Mục tiêu

Huấn luyện Qwen để:

1. Gọi tool đúng format trong multi-turn.
2. Dùng kết quả tool để trả lời cuối cùng.
3. Tối ưu policy qua PPO-like update (`adv_estimator=gae` hoặc `grpo` biến thể).

## 2. Khung config gợi ý (PPO-style)

```yaml
# qwen_ppo_toolcall_multiturn.yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  train_files: ~/data/toolcall/train.parquet
  val_files: ~/data/toolcall/val.parquet
  return_raw_chat: true
  max_prompt_length: 2048
  max_response_length: 1024
  train_batch_size: 128

algorithm:
  adv_estimator: gae
  use_kl_in_reward: false

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
    kl_loss_type: low_var_kl
  rollout:
    name: sglang
    n: 4
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.8
    multi_turn:
      enable: true
      tool_config_path: /ABS_PATH/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml
      max_assistant_turns: 5

critic:
  strategy: fsdp2
  model:
    path: Qwen/Qwen2.5-3B-Instruct
  ppo_micro_batch_size_per_gpu: 8

trainer:
  n_gpus_per_node: 8
  nnodes: 1
  logger: [console, wandb]
  project_name: qwen-ppo-toolcall
  experiment_name: multiturn-v1
  test_freq: 20
  save_freq: 50
  total_epochs: 5
```

## 3. Lệnh chạy

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=/ABS_PATH \
  --config-name=qwen_ppo_toolcall_multiturn
```

## 4. Reward function cho tool-calling

Nên dùng custom reward với các term:

- `+1`: kết quả cuối đúng.
- `+0.2`: tool call JSON valid.
- `-0.1`: mỗi tool call dư.
- `-1`: tool hallucination hoặc call sai schema.

## 5. Metric nên theo dõi

1. `task_success_rate`
2. `tool_parse_success_rate`
3. `avg_tool_calls`
4. `actor/ppo_kl`, `actor/pg_clipfrac`
5. `timing/gen`, `timing/update_actor`

## 6. Pitfalls

1. Đặt `max_prompt_length` quá thấp làm mất history turn.
2. Dùng template không khớp tool schema.
3. Chỉ tối ưu task reward mà bỏ qua formatting reward.
