# 05 - End-to-End Training Runbook

## 1. Chuẩn bị

1. Cài `verl` + deps đúng theo docs.
2. Chuẩn bị dataset/tool config/reward script.
3. Xác nhận cluster resources và storage path checkpoint.

## 2. Lệnh chạy mẫu

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=/abs/path/config \
  --config-name=toolcall_multiturn_grpo \
  reward.custom_reward_function.path=/abs/path/reward.py \
  reward.custom_reward_function.name=compute_score
```

## 3. Monitoring tối thiểu

- `timing/gen`, `timing/update_actor`, `timing/update_critic`
- `actor/ppo_kl`, `actor/pg_loss`
- task success trên val
- tool parse success rate

## 4. Checkpoint strategy

- Save mỗi `N` steps + retention policy.
- Khi resume: khóa commit config + seed để tái lập.

## 5. Done Criteria

- Chạy qua ít nhất 1 epoch.
- Val metrics cải thiện so với mốc khởi đầu.
- Checkpoint restore chạy được.
