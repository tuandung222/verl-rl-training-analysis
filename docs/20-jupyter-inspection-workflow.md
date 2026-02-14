# Dùng `verl` trong Jupyter để inspect từng component

Có thể làm được. `verl` là Python package, không bắt buộc phải chạy chỉ qua CLI.

## 1. Mục tiêu notebook workflow

1. Nạp config Hydra/OmegaConf.
2. Khởi tạo dataset/tokenizer riêng để inspect.
3. Gọi trực tiếp các hàm lõi như advantage/loss.
4. Chạy một step mini offline để debug signal.

## 2. Ví dụ notebook snippets

### 2.1 Nạp config

```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("/ABS_PATH/verl/verl/trainer/config/ppo_trainer.yaml")
print(cfg.algorithm)
```

### 2.2 Inspect dataset

```python
from verl.utils import hf_tokenizer
from verl.trainer.main_ppo import create_rl_dataset

model_path = "Qwen/Qwen2.5-0.5B-Instruct"
tok = hf_tokenizer(model_path, trust_remote_code=False)

ds = create_rl_dataset(
    data_paths=["/ABS_PATH/train.parquet"],
    data_config=cfg.data,
    tokenizer=tok,
    processor=None,
    is_train=True,
    max_samples=32,
)

sample = ds[0]
print(sample.keys())
```

### 2.3 Inspect advantage function

```python
import torch
import numpy as np
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

token_level_rewards = torch.randn(8, 16)
response_mask = torch.ones(8, 16)
index = np.array([0,0,1,1,2,2,3,3])

adv, ret = compute_grpo_outcome_advantage(
    token_level_rewards=token_level_rewards,
    response_mask=response_mask,
    index=index,
)
print(adv.shape, ret.shape)
```

### 2.4 Inspect policy loss

```python
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

old_log_prob = torch.randn(4, 10)
log_prob = old_log_prob + 0.01 * torch.randn(4, 10)
advantages = torch.randn(4, 10)
mask = torch.ones(4, 10)

loss, metrics = compute_policy_loss_vanilla(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    response_mask=mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
)
print(loss.item(), metrics)
```

## 3. Khi nào vẫn nên dùng CLI

- Khi cần full distributed run với Ray cluster.
- Khi cần reproducible production jobs.
- Khi cần checkpoint/logging flow đầy đủ.

## 4. Hybrid workflow khuyến nghị

1. Notebook cho inspect và unit-debug từng stage.
2. Script/CLI cho training thật.
3. Dùng notebook để phân tích artifacts sau mỗi run (metrics, traces, failed samples).

## 5. Một notebook stack gợi ý

- `00_data_sanity.ipynb`
- `01_reward_debug.ipynb`
- `02_advantage_debug.ipynb`
- `03_loss_debug.ipynb`
- `04_run_diagnostics.ipynb`

Cách này giúp Data Scientist làm việc từng lớp thay vì debug cả pipeline lớn ngay từ đầu.
