# Getting Started (thực dụng)

## 0. Mục tiêu mini

Chạy được PPO GSM8K 1 GPU, log ra metric hợp lệ, hiểu các knob cơ bản.

## 1. Chuẩn bị

- GPU >= 24GB (khuyến nghị theo docs quickstart)
- Cài `verl` theo `docs/start/install.rst`
- Chuẩn bị dataset parquet:

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

## 2. Lệnh chạy baseline PPO

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  critic.optim.lr=1e-5 \
  critic.ppo_micro_batch_size_per_gpu=4 \
  trainer.logger=console \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=15
```

## 3. Checkpoint + merge

Checkpoint mặc định ở:

- `checkpoints/${trainer.project_name}/${trainer.experiment_name}`

Merge về HF format:

```bash
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir <path_to_actor_ckpt> \
  --target_dir <output_hf_dir>
```

## 4. Debug checklist

- OOM khi rollout: giảm `rollout.gpu_memory_utilization`, giảm response length.
- OOM khi update: giảm `actor/critic ppo_micro_batch_size_per_gpu`.
- Không tiến bộ metric: kiểm tra reward fn, prompt template, KL coef.
- Throughput thấp: đọc `docs/perf/perf_tuning.*` và ưu tiên profile theo step.
