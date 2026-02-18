# 07 - Performance Tuning and Scale-Out

## 1. Thứ tự tuning khuyến nghị

1. Sống sót qua OOM.
2. Tăng throughput rollout.
3. Tối ưu learning dynamics.

## 2. OOM playbook

- Giảm `ppo_micro_batch_size_per_gpu`.
- Giảm `rollout.n`.
- Giảm `max_prompt_length`/`max_response_length` có kiểm soát.
- Bật `enable_gradient_checkpointing`.

## 3. Throughput playbook

- Tối ưu `rollout.gpu_memory_utilization`.
- Chọn đúng TP/DP rollout.
- Bật async/server mode khi workload tool latency cao.

## 4. Scale-out

- Tăng `nnodes` từng bước; giữ benchmark chuẩn.
- Theo dõi communication overhead khi tăng parallelism.
- Cố định workload chuẩn để so apples-to-apples.

## 5. Done Criteria

- Có profile trước/sau tuning.
- Throughput cải thiện rõ mà không làm xấu quality.
- Multi-node run ổn định.
