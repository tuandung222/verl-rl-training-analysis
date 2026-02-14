# LLM for Planning & Orchestration với verl

## Vì sao liên quan

Các hệ thống agent/planning cần model:

- biết chia nhỏ nhiệm vụ,
- gọi tool đúng lúc,
- giữ trạng thái multi-turn,
- tối ưu reward dài hạn theo outcome.

`verl` hỗ trợ nền tảng này qua multi-turn rollout + tool integration.

## Các điểm cần đọc trong upstream

- `docs/start/agentic_rl.rst`
- `docs/sglang_multiturn/*`
- `examples/tutorial/agent_loop_get_started/*`
- `verl/workers/rollout/sglang_rollout/*`
- `verl/interactions/*`

## Concept map cho planning RL

1. Dataset phải encode được traces đa lượt (user/tool/assistant).
2. Reward nên kết hợp:
   - outcome correctness,
   - tool efficiency/cost,
   - safety/format compliance.
3. Rollout engine cần ổn định cho multi-turn và tokenization template.
4. Cần observability tốt để truy vết lỗi orchestration theo turn.

## Hướng build curriculum

1. Single-turn task correctness.
2. Multi-turn không tool.
3. Multi-turn có tool deterministic.
4. Multi-turn tool-use noisy environment.
5. Mixed-task training với reward decomposition.

## Chỉ số nên theo dõi thêm

- Tool call success rate.
- Plan completion rate.
- Average turn count per solved task.
- Cost-aware reward (tokens + tool latency).
