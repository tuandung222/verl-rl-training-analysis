# Operational Playbook để master verl

## 1. Quy trình chuẩn khi bắt đầu experiment

1. Khóa một cấu hình baseline chạy được end-to-end.
2. Chạy 100-300 step sanity trước khi mở rộng quy mô.
3. Chỉ thay đổi 1 nhóm biến mỗi lần (reward, rollout, optimizer, batching...).
4. Lưu run sheet: config diff + metric curve + failure notes.

## 2. Nhóm biến cần theo dõi

- Stability:
  - `actor/ppo_kl`, `actor/pg_clipfrac`, `critic/vf_loss`, reward mean/std.
- Efficiency:
  - `timing/gen`, `timing/update_actor`, `timing/update_critic`, tokens/s.
- Data quality:
  - response length, invalid outputs, reward sparsity.

## 3. Quy tắc tuning ưu tiên

1. Giải quyết OOM trước (micro-batch, sequence length).
2. Giải quyết throughput sau (rollout engine + placement).
3. Tối ưu learning dynamics cuối cùng (lr, kl_coef, estimator, reward shaping).

## 4. Failure modes phổ biến

- Reward hacking: score tăng nhưng quality thực giảm.
- KL collapse hoặc drift quá mạnh.
- Critic learning lag làm advantage nhiễu.
- Training-inference mismatch gây policy update sai hướng.

## 5. Mở rộng thuật toán

`verl` thuận tiện mở rộng vì:

- Advantage estimator ở `core_algos` có thể cắm thêm.
- Dataflow trong trainer rõ ràng theo từng stage.
- Config tree Hydra cho phép tạo preset thuật toán mới.

Chiến lược mở rộng an toàn:

1. Fork từ pipeline PPO chạy ổn.
2. Đổi 1 điểm thuật toán (vd: advantage hoặc KL control).
3. Viết test tối thiểu cho regression.
4. So sánh với baseline bằng cùng seed/config compute budget.
