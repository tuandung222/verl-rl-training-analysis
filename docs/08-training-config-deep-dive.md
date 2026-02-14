# Phân tích chi tiết training config của `verl`

Tài liệu này phân tích trực tiếp từ:

- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/actor/actor.yaml`
- `verl/trainer/config/critic/critic.yaml`
- `verl/trainer/config/rollout/rollout.yaml`
- `verl/trainer/config/reward/reward.yaml`
- `verl/trainer/config/data/legacy_data.yaml`

## 1. Tư duy config của `verl`

`verl` dùng Hydra, cấu hình theo cây gồm 6 khối chính:

1. `data`: dữ liệu + format + batching ở mức prompt.
2. `actor_rollout_ref`: policy model, rollout engine, reference policy.
3. `critic`: value model và optimizer cho critic.
4. `reward`: reward function / reward model.
5. `algorithm`: objective-level knobs (adv estimator, KL, rollout correction).
6. `trainer`: orchestration-level knobs (cluster, logging, checkpoint, loop).

Nguyên tắc quan trọng:

- `algorithm` quyết định bản chất học (GAE, GRPO, KL...).
- `actor/critic/rollout` quyết định tính khả thi hệ thống (OOM, throughput, latency).
- `trainer` quyết định vòng đời run (save/resume/val/log).

## 2. Khối `data`

Các khóa quan trọng:

- `train_files`, `val_files`: parquet/json input.
- `prompt_key`, `reward_fn_key`: cột nào của dataset dùng làm prompt/reward routing.
- `max_prompt_length`, `max_response_length`: ảnh hưởng trực tiếp memory + speed.
- `train_batch_size`: số prompt toàn cục mỗi iteration rollout.
- `return_raw_chat`: cực quan trọng cho multi-turn/tool-calling.
- `filter_overlong_prompts`, `truncation`: chiến lược xử lý prompt dài.
- `sampler.*`: cắm curriculum/custom sampler.

Quy tắc thực dụng:

1. RL tool-calling multiturn thường cần `return_raw_chat=True` để giữ hội thoại dạng message.
2. Trước khi scale, bật `filter_overlong_prompts=True` để tránh outlier gây OOM.
3. `train_batch_size` tăng quá nhanh sẽ đẩy áp lực lên rollout engine trước khi actor update nghẽn.

## 3. Khối `actor_rollout_ref.model`

Mục tiêu: mô tả model và behavior trong training engine.

Các trường cốt lõi:

- `path`, `tokenizer_path`, `trust_remote_code`, `override_config`.
- `enable_gradient_checkpointing`, `enable_activation_offload`.
- `use_remove_padding`, `use_fused_kernels`, `tiled_mlp`.
- `lora_rank`, `lora_alpha`, `target_modules`, `lora_adapter_path`.

Heuristics:

- Nếu VRAM chật: ưu tiên `gradient_checkpointing=True`, sau đó xét `activation_offload`.
- Nếu sequence dài: `use_remove_padding=True` thường cải thiện đáng kể hiệu năng.
- Nếu train adapter: kiểm tra logic ref policy, vì một số mode có thể fuse ref-in-actor.

## 4. Khối `actor`

Đây là phần nhạy nhất với stability.

### Batching

- `ppo_mini_batch_size`: mini-batch toàn cục cho update.
- `ppo_micro_batch_size_per_gpu`: micro-batch local chống OOM.
- `use_dynamic_bsz`, `ppo_max_token_len_per_gpu`: sequence packing theo token budget.

### Policy loss

- `policy_loss.loss_mode`: `vanilla`, `gpg`, `clip_cov`, `kl_cov`, `cispo`, `geo_mean`, `bypass_mode`...
- `clip_ratio`, `clip_ratio_low/high`, `clip_ratio_c`: clip parameters.
- `loss_agg_mode`: token/sequence aggregation mode.
- `entropy_coeff`: entropy regularization.

### KL controls

- `use_kl_loss`, `kl_loss_coef`, `kl_loss_type`.

### Optimization

- `optim.lr`, warmup params, weight decay.
- `ppo_epochs`, `shuffle`.

Practical order khi tuning:

1. Chỉnh micro-batch để qua OOM.
2. Chỉnh `clip_ratio`, KL coef để ổn định.
3. Chỉnh lr/warmup.
4. Mới đụng loss mode nâng cao.

## 5. Khối `rollout`

`rollout` là engine inference riêng (vLLM/SGLang/HF/TRTLLM) dùng để generate.

Các khóa cốt lõi:

- `name`: engine (`sglang`, `vllm`, ...).
- `gpu_memory_utilization`: mức dùng HBM cho KV cache.
- `tensor_model_parallel_size`, `data_parallel_size`, `expert_parallel_size`.
- `max_num_batched_tokens`, `max_model_len`, `max_num_seqs`.
- `n`: số response mỗi prompt (quan trọng cho GRPO/RLOO).
- `calculate_log_probs`: cần cho một số mode correction/debug.

### Multi-turn / tool-calling

- `multi_turn.enable`
- `multi_turn.tool_config_path`
- `multi_turn.max_assistant_turns`, `max_user_turns`
- `multi_turn.format`
- `multi_turn.tokenization_sanity_check_mode`

Lưu ý:

- Multi-turn nên dùng `rollout.name=sglang` theo examples upstream.
- `n>1` + multi-turn làm token budget tăng rất nhanh.

## 6. Khối `ref`

Dùng để tính logprob của policy tham chiếu cho KL reward/loss.

Trường quan trọng:

- `log_prob_micro_batch_size_per_gpu`
- `log_prob_use_dynamic_bsz`
- `log_prob_max_token_len_per_gpu`

Nếu KL bật mà ref cấu hình yếu, throughput sẽ tụt mạnh.

## 7. Khối `critic`

Chỉ bắt buộc với estimator cần value function (điển hình GAE/PPO).

Các trường chính:

- `enable` (null thì auto theo estimator)
- `model.path`
- `ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu`
- `cliprange_value`
- `loss_agg_mode`

Rule of thumb:

- Nếu critic loss dao động cực lớn, giảm lr critic trước khi giảm lr actor.

## 8. Khối `reward`

Hai mode chính:

1. Function-based reward (`custom_reward_function`).
2. Reward model (`reward_model.enable=True`).

Trường then chốt:

- `custom_reward_function.path/name`
- `reward_model.enable`
- `reward_model.enable_resource_pool`
- `reward_model.rollout.*`

Cho tool-calling, function-based reward rất phổ biến: chấm đúng/sai format tool call + execution result.

## 9. Khối `algorithm`

Đây là lõi thuật toán.

- `adv_estimator`: `gae`, `grpo`, `rloo`, `gpg`, `remax`, `opo`, ...
- `gamma`, `lam`
- `use_kl_in_reward`
- `kl_penalty`, `kl_ctrl.type/kl_coef/horizon/target_kl`
- `rollout_correction.*` (từ default config rollout_correction)

Quy tắc mapping nhanh:

- PPO cổ điển: `adv_estimator=gae`, critic bật.
- GRPO: `adv_estimator=grpo`, thường `rollout.n > 1`, có thể dùng KL loss.

## 10. Khối `trainer`

Điều phối vòng đời chạy.

- `nnodes`, `n_gpus_per_node`
- `total_epochs` hoặc `total_training_steps`
- `save_freq`, `test_freq`, `val_before_train`
- `resume_mode`, `resume_from_path`
- `logger`, `project_name`, `experiment_name`
- `use_legacy_worker_impl`

Checklist trước khi chạy thật:

1. `project_name/experiment_name` rõ ràng.
2. `save_freq/test_freq` phù hợp budget.
3. `resume_mode` đúng với trạng thái job.

## 11. Config interactions dễ gây lỗi

1. `rollout.n` tăng nhưng không tăng token budget -> OOM rollout.
2. Bật KL (`use_kl_loss` hoặc `use_kl_in_reward`) nhưng ref worker batch quá nhỏ -> nghẽn.
3. `return_raw_chat=False` cho task tool-calling -> mất cấu trúc hội thoại.
4. `max_prompt_length` thấp + truncation không phù hợp -> reward sai do cắt mất tool schema.
5. `ppo_mini_batch_size` không khớp cluster size -> accumulation không như kỳ vọng.

## 12. Minimal config templates

### PPO baseline (single-turn)

```yaml
algorithm:
  adv_estimator: gae
  use_kl_in_reward: false

actor_rollout_ref:
  rollout:
    name: vllm
    n: 1

critic:
  enable: true
```

### GRPO multiturn tool-calling

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_coef: 0.001
  rollout:
    name: sglang
    n: 8
    multi_turn:
      enable: true
      tool_config_path: /path/to/tools.yaml

data:
  return_raw_chat: true
```
