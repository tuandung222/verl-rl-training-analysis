# `verl` hỗ trợ những thuật toán RL nào?

## 1. Theo mã nguồn core `verl`

Trong `verl/trainer/ppo/core_algos.py`, thấy rõ registry advantage/policy loss cho các nhánh sau:

### Advantage estimators

- `gae`
- `grpo`
- `grpo_vectorized`
- `grpo_passk`
- `reinforce_plus_plus`
- `reinforce_plus_plus_baseline`
- `rloo`
- `rloo_vectorized`
- `remax`
- `opo`
- `gpg`
- `optimal_token_baseline`
- `tir_optimal_token_baseline`

### Policy loss modes (điển hình)

- `vanilla` (PPO clip)
- `gspo`
- `sapo`
- `gpg`
- `clip_cov`
- `kl_cov`
- `geo_mean`
- `cispo`
- `bypass_mode` (cho rollout correction settings)

## 2. Theo examples

Có các trainer/examples thư mục riêng:

- `examples/ppo_trainer`
- `examples/grpo_trainer`
- `examples/gspo_trainer`
- `examples/gpg_trainer`
- `examples/remax_trainer`
- `examples/rloo_trainer`
- `examples/reinforce_plus_plus_trainer`
- `examples/sapo_trainer`
- `examples/cispo_trainer`
- `examples/otb_trainer`

## 3. Theo recipe submodule

`recipe/` mở rộng thêm nhiều recipe như:

- DAPO
- SPIN (online DPO flavor)
- SPPO
- PRIME
- FlowRL
- nhiều recipe domain-specific khác

## 4. Thuật toán KHÔNG thấy first-class trainer trong core snapshot

- DPO (core trainer chuẩn): chưa thấy.
- KTO: chưa thấy.
- ORPO: chưa thấy.

## 5. Kết luận

`verl` hiện mạnh ở họ PPO-like online RL + nhiều biến thể estimator/loss; hệ recipe mở rộng thêm alignment algorithms. Với DPO/KTO/ORPO, cần xem theo ecosystem hoặc tự mở rộng theo pattern framework.
