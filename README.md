# verl RL Training Analysis Handbook

Repository này dành cho người mới học RL training cho LLM, đặc biệt với `verl` (Volcano Engine Reinforcement Learning for LLMs).

## Mục tiêu

- Giải thích kiến trúc và luồng chạy thật của `verl` từ mã nguồn gốc.
- Làm rõ các concept quan trọng: PPO/GRPO, actor-critic, rollout, reward, KL control, scaling.
- Cung cấp playbook thực hành để tự chạy, debug, tối ưu và mở rộng.
- Hỗ trợ ngữ cảnh `LLM for Planning & Orchestration` (agent loop, multi-turn, tool-use).

## Nguồn gốc phân tích

- Upstream: `https://github.com/verl-project/verl`
- Snapshot đã phân tích: commit `395938b2` (branch `main`)
- Tài liệu + code map được tổng hợp từ:
  - `README.md`
  - `docs/start/*`
  - `docs/workers/*`
  - `docs/algo/*`
  - `verl/trainer/main_ppo.py`
  - `verl/trainer/ppo/ray_trainer.py`
  - `verl/utils/dataset/rl_dataset.py`
  - `verl/workers/*`

## Mục lục

1. [Architecture Map](./docs/01-architecture-map.md)
2. [Code Walkthrough](./docs/02-code-walkthrough.md)
3. [Core Concepts](./docs/03-core-concepts.md)
4. [Quickstart học nhanh](./docs/04-getting-started.md)
5. [Operational Playbook](./docs/05-operational-playbook.md)
6. [LLM Planning & Orchestration](./docs/06-planning-orchestration.md)
7. [Learning Path 30 ngày](./docs/07-learning-path.md)
8. [Tài nguyên upstream](./references/upstream-links.md)
9. [External Framework Links](./references/external-frameworks-links.md)

10. [Training Config Deep Dive](./docs/08-training-config-deep-dive.md)
11. [Architecture vs Ecosystem](./docs/09-architecture-vs-ecosystem.md)
12. [RL Framework Blueprint](./docs/10-rl-framework-blueprint.md)
13. [LLaMA-Factory Tool-Calling RL Checklist](./docs/11-llamafactory-toolcalling-rl-checklist.md)
14. [Example: Qwen + PPO Tool-Calling Multiturn](./docs/12-example-qwen-ppo-toolcalling-multiturn.md)
15. [Example: Gemma + DPO Tool-Calling Multiturn](./docs/13-example-gemma-dpo-toolcalling-multiturn.md)
16. [Example: GPT-OSS + GRPO Tool-Calling Multiturn](./docs/14-example-gpt-oss-grpo-toolcalling-multiturn.md)
17. [PPO from verl code](./docs/15-algo-ppo-from-verl-code.md)
18. [GRPO from verl code](./docs/16-algo-grpo-from-verl-code.md)
19. [DPO & KTO in verl ecosystem](./docs/17-algo-dpo-kto-from-verl-ecosystem.md)
20. [ORPO in verl ecosystem](./docs/18-algo-orpo-from-verl-ecosystem.md)
21. [Supported RL Algorithms Map](./docs/19-supported-rl-algorithms-map.md)
22. [Jupyter Inspection Workflow](./docs/20-jupyter-inspection-workflow.md)


## Đối tượng đọc

- Người mới vào RLHF/RLVR cho LLM.
- Kỹ sư ML muốn chuyển từ SFT sang RL post-training.
- Team xây agent loop/tool-use cần nền tảng RL training có thể mở rộng.

## Cách dùng repo này

- Đọc theo thứ tự 1 -> 7 nếu mới bắt đầu.
- Nếu đã có nền tảng, bắt đầu từ:
  - Kiến trúc: `docs/01-architecture-map.md`
  - Vận hành thực tế: `docs/05-operational-playbook.md`
  - Agentic RL: `docs/06-planning-orchestration.md`
