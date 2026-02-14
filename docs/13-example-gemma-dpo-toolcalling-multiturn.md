# Ví dụ: Gemma + DPO cho tool-calling multiturn

## 0. Ghi chú phạm vi

Trong `verl` core hiện tại, DPO không phải trainer chính thức dạng `main_dpo.py` như PPO.

Các đường triển khai thực tế:

1. Dùng `verl` ecosystem recipe (đặc biệt hướng Online-DPO qua `recipe/spin`).
2. Hoặc dùng TRL/LLaMA-Factory để chạy DPO offline trên preference data đa lượt.

Tài liệu này đưa ví dụ thực dụng qua LLaMA-Factory vì đường setup đơn giản cho DPO.

## 1. Data format cho multiturn tool-calling DPO

Mỗi sample cần:

- `chosen`: transcript tốt (tool calls hợp lệ, outcome đúng)
- `rejected`: transcript xấu (tool sai hoặc outcome sai)
- cùng prompt context

## 2. YAML ví dụ (LLaMA-Factory)

```yaml
# gemma_dpo_toolcall_multiturn.yaml
model_name_or_path: google/gemma-2-9b-it
trust_remote_code: true

stage: dpo
do_train: true
finetuning_type: lora

template: gemma

dataset: toolcall_pref_multiturn
cutoff_len: 4096
max_samples: 200000
overwrite_cache: true
preprocessing_num_workers: 16

dataloader_num_workers: 4
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5e-6
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true

pref_beta: 0.1
pref_loss: sigmoid
dpo_label_smoothing: 0.0

# nếu cần reference tách riêng
ref_model: google/gemma-2-9b-it

output_dir: saves/gemma2-9b/dpo-toolcall
logging_steps: 10
save_steps: 500
plot_loss: true
```

## 3. Ý tưởng tạo preference pair

1. Lấy trajectory từ policy hiện tại + tool execution logs.
2. Đặt scorer chấm theo multi-criteria.
3. Ghép pair theo cùng prompt.
4. Lọc những cặp chênh lệch điểm quá nhỏ.

## 4. Nếu muốn giữ trong hệ `verl`

Có thể nghiên cứu `recipe/spin`:

- `recipe/spin/main_spin.py`
- `recipe/spin/spin_trainer.py`
- `recipe/spin/dp_actor.py`

Đây là online DPO flavor, không phải drop-in thay cho `main_ppo.py`.
