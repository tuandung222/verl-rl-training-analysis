# DPO & KTO trong bối cảnh mã nguồn `verl` hiện tại

## 1. Trạng thái hỗ trợ thực tế

### Trong `verl` core (`verl/`)

- Không có entrypoint chuẩn kiểu `main_dpo.py` hoặc `main_kto.py` tương đương `main_ppo.py`.
- Core tập trung vào PPO-like runtime với nhiều estimator/loss variants.

### Trong hệ `verl` ecosystem

- Có hướng DPO online qua `recipe/spin`:
  - `recipe/spin/main_spin.py`
  - `recipe/spin/spin_trainer.py`
  - `recipe/spin/dp_actor.py`

KTO:

- Không thấy implementation KTO first-class trong `verl` core/recipe hiện snapshot này.

## 2. DPO là gì (góc toán)

DPO dùng dữ liệu preference `(x, y+, y-)` để học trực tiếp policy mà không cần reward model explicit runtime.

Ý tưởng cốt lõi:

- So sánh log-ratio giữa chosen vs rejected, có tham chiếu reference policy.
- Tối ưu logistic objective để policy tăng xác suất cho chosen hơn rejected.

## 3. Online DPO trong `recipe/spin`

SPIN thực hiện iterative self-play preference optimization:

1. Rollout từ policy hiện tại.
2. Sinh preference labels theo reward/outcome.
3. Update actor theo DPO-style objective.
4. Có thể cập nhật ref policy định kỳ.

## 4. KTO là gì

KTO (Kahneman-Tversky Optimization) dùng nhãn desirable/undesirable thay vì pair chosen/rejected đầy đủ.

Thường phù hợp khi dữ liệu preference pair khó thu thập đồng bộ.

## 5. Đề xuất nếu bạn cần DPO/KTO production

1. Nếu muốn giữ runtime scale-out của `verl`, cần đầu tư một trainer riêng theo pattern `main_ppo` + custom loss flow.
2. Nếu cần triển khai nhanh DPO/KTO, dùng TRL hoặc LLaMA-Factory/Axolotl (vì có trainer/method sẵn).

## 6. Kết luận

- DPO: có đường đi trong ecosystem `recipe/spin` (online flavor), không phải core trainer chuẩn.
- KTO: chưa thấy hỗ trợ first-class trong snapshot code này.
