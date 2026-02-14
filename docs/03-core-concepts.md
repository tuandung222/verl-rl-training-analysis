# Core Concepts cho người mới RL training LLM

## 1. Policy, Rollout, Trajectory

- Policy (actor): mô hình đang train.
- Rollout: chạy generate response từ prompt.
- Trajectory: prompt -> tokens -> reward/value/logprob.

Trong LLM RL, "state/action" được map sang token sequence.

## 2. Actor-Critic vs GRPO style

- PPO chuẩn cần critic để ước lượng value và tính advantage ổn định (GAE).
- GRPO/RLOO có thể giảm hoặc bỏ vai trò critic, dùng group/statistical baselines.
- `verl` hỗ trợ nhiều estimator để bạn thử trade-off giữa ổn định và chi phí compute.

## 3. KL control

Mục tiêu: tránh policy drift quá xa model tham chiếu.

Hai cách chính:

- KL trong loss actor.
- KL penalty trong reward (`algorithm.use_kl_in_reward=True`).

KL control rất quan trọng để giảm collapse hoặc hành vi bất thường sau vài nghìn step.

## 4. Batch size, mini-batch, micro-batch

- `train_batch_size`: số prompt dùng để rollout mỗi step.
- `ppo_mini_batch_size`: batch cho update actor/critic.
- `*_micro_batch_size*`: chunk nhỏ để fit memory GPU.

`micro_batch` là nút điều khiển OOM phổ biến nhất.

## 5. Training-inference mismatch

Rollout engine (vLLM/SGLang) và training engine (FSDP/Megatron) có thể khác nhau.

Mismatch này gây off-policy drift; vì vậy `verl` bổ sung rollout correction (IS/RS/bypass modes).

## 6. Why verl cho production

- Scale out theo Ray worker groups.
- Backend linh hoạt (FSDP/Megatron + vLLM/SGLang).
- Dễ mở rộng thuật toán nhờ controller dataflow + config-driven.
- Tập trung vào throughput lẫn ổn định huấn luyện.
