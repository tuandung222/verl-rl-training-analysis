# Code Walkthrough (đi từ code thật)

## 1. `main_ppo.py`

Mốc đọc quan trọng:

- `run_ppo(config)`:
  - Tạo `ray.runtime_env` và gọi `ray.init(...)`.
  - Tạo remote `TaskRunner`.
- `TaskRunner.run(config)`:
  - Validate config (có/không critic, ref policy).
  - Resolve model path local (copy from remote fs nếu cần).
  - Tạo tokenizer/processor.
  - Tạo dataset + sampler.
  - Init `RayPPOTrainer` rồi chạy `init_workers()` và `fit()`.

## 2. `RayPPOTrainer`

Các phần chính:

- `_create_dataloader(...)`
  - Dựa vào `create_rl_dataset` và `create_rl_sampler`.
  - Dataset class lấy từ `get_dataset_class(data_config)` -> dễ mở rộng custom data.
- `fit()`
  - Vòng epoch/step.
  - Điều phối generate, reward, advantage, update actor/critic.
  - Log metric, validate, save checkpoint.

## 3. Reward + KL + Advantage

- `apply_kl_penalty(data, kl_ctrl, kl_penalty)`:
  - Lấy `old_log_probs` và `ref_log_prob` để tính KL.
  - Trừ KL penalty trực tiếp vào token-level rewards.
- `compute_advantage(...)`:
  - Hỗ trợ nhiều estimator: `gae`, `grpo`, `reinforce_plus_plus`, `rloo`, ...
  - Đây là điểm cốt lõi để mở rộng thuật toán mà không phá cấu trúc trainer.

## 4. Dataset layer

`verl/utils/dataset/rl_dataset.py`:

- Input parquet/json -> HF dataset.
- Apply chat template.
- Optional filter prompt quá dài.
- Optional multimodal (image/video) qua processor.
- `collate_fn` tách tensor vs non-tensor rõ ràng.

## 5. Worker layer

Thư mục `verl/workers/` phân role rõ:

- `actor/`, `critic/`, `rollout/`, `reward_manager/`, `engine/`, `sharding_manager/`.
- `engine/` tách backend-specific implementation (fsdp/megatron/veomni/...).
- `engine_workers.py` + `fsdp_workers.py` + `megatron_workers.py`: adapter từ trainer API sang backend cụ thể.

## 6. Config system

`verl/trainer/config/ppo_trainer.yaml` dùng Hydra defaults tree:

- `actor_rollout_ref.*`
- `critic.*`
- `reward.*`
- `algorithm.*`
- `trainer.*`

Tư duy config của `verl`:

- Tách rõ algorithmic config và systems config.
- Có nhiều cờ tuning cho scaling/perf nhưng giữ entrypoint thống nhất.
