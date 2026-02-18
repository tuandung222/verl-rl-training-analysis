# Series: Dùng `verl` Train Agent Tool-Calling Multi-Turn

Series này là playbook end-to-end để train LLM Agent bằng `verl` cho bài toán tool-calling đa lượt.

## Ai nên đọc

- ML Engineer/Research Engineer bắt đầu RL post-training cho agent.
- Team muốn chuyển từ single-turn sang multi-turn tool-use.
- Data Scientist cần framework rõ ràng để debug và cải thiện policy.

## Lộ trình đọc

1. [01 - Problem Framing & System Design](./01-problem-framing-and-system-design.md)
2. [02 - Data Pipeline & Dataset Spec](./02-data-pipeline-and-dataset-spec.md)
3. [03 - Reward Design for Tool-Calling](./03-reward-design-for-tool-calling.md)
4. [04 - verl Training Config Blueprint](./04-verl-training-config-blueprint.md)
5. [05 - End-to-End Training Runbook](./05-end-to-end-training-runbook.md)
6. [06 - Evaluation, Debugging, and Iteration](./06-evaluation-debugging-iteration.md)
7. [07 - Performance Tuning and Scale-Out](./07-performance-tuning-and-scale-out.md)
8. [08 - Production Checklist](./08-production-checklist.md)

## Kết quả kỳ vọng

- Chuẩn bị đúng dữ liệu multi-turn + tool schema.
- Viết được reward function có thể tối ưu hành vi tool-use.
- Chạy được training GRPO/PPO-like với `verl` cho agent loop.
- Có quy trình đo lường và debug rõ ràng theo failure modes.
