# Learning Path 30 ngày

## Tuần 1: Nắm kiến trúc + chạy baseline

- Đọc `01`, `02`, `03`.
- Chạy GSM8K PPO 1 GPU.
- Mục tiêu: hiểu end-to-end log và checkpoint.

## Tuần 2: Tuning có hệ thống

- Đọc `05`.
- Làm 3 sweep nhỏ:
  - micro-batch,
  - KL coef,
  - rollout memory utilization.
- Mục tiêu: biết xử lý OOM + nâng throughput.

## Tuần 3: Thuật toán và reward

- Đọc `docs/algo/*` upstream: `ppo`, `grpo`, `dapo`, `rollout_corr`.
- So sánh ít nhất 2 estimator trên cùng task.
- Mục tiêu: hiểu trade-off stability/sample-efficiency.

## Tuần 4: Agentic RL / Planning

- Đọc `06` + `docs/start/agentic_rl.rst` upstream.
- Làm bài tập mini: task tool-use 2-3 turn với reward đơn giản.
- Mục tiêu: có một baseline orchestration RL pipeline tái lập.
