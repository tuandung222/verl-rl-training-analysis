# Ví dụ: GPT-OSS + GRPO cho tool-calling multiturn

## 0. Model path và lưu ý

Theo tài liệu GPT-OSS của LLaMA-Factory, model ví dụ là `openai/gpt-oss-20b`.

Nguồn: https://llamafactory.readthedocs.io/en/latest/advanced/best_practice/gpt-oss.html

Ví dụ dưới đây dùng `verl` với mục tiêu GRPO multiturn tool-calling.

## 1. Config gợi ý (verl)

```yaml
# gpt_oss_grpo_toolcall_multiturn.yaml
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
  max_prompt_length: 4096
  max_response_length: 1536
  train_batch_size: 64
  filter_overlong_prompts: true

algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
  use_kl_in_reward: false

actor_rollout_ref:
  model:
    path: openai/gpt-oss-20b
    trust_remote_code: true
    use_remove_padding: true
    enable_gradient_checkpointing: true
  actor:
    strategy: fsdp2
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 2
    use_kl_loss: true
    kl_loss_coef: 0.001
  rollout:
    name: sglang
    n: 8
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.85
    multi_turn:
      enable: true
      tool_config_path: /ABS_PATH/tool_config.yaml
      max_assistant_turns: 6
      format: hermes

critic:
  enable: false

trainer:
  n_gpus_per_node: 8
  nnodes: 1
  project_name: gpt-oss-grpo-toolcall
  experiment_name: mt-v1
  logger: [console, wandb]
  test_freq: 20
  save_freq: 40
  total_epochs: 3
```

## 2. Lệnh chạy

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=/ABS_PATH \
  --config-name=gpt_oss_grpo_toolcall_multiturn
```

## 3. Tuning ưu tiên cho model lớn

1. Giảm `rollout.n` trước khi giảm max lengths.
2. Kiểm soát `gpu_memory_utilization` của rollout engine.
3. Dùng `ppo_micro_batch_size_per_gpu` rất nhỏ (1-2) để tránh OOM update.
4. Theo dõi latency tail ở multi-turn tool execution.

## 4. Metrics bổ sung cho GPT-OSS use case

- tool completion success
- final answer correctness
- avg tokens/episode
- avg turns/episode
- wall clock per step
