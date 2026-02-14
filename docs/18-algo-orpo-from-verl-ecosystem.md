# ORPO trong bối cảnh `verl`

## 1. Trạng thái trong `verl` snapshot hiện tại

- Không thấy implementation ORPO trainer first-class trong `verl` core.
- Không thấy recipe ORPO riêng trong submodule `recipe` ở snapshot đã kiểm tra.

## 2. ORPO là gì (tổng quan)

ORPO là một phương pháp preference optimization kết hợp supervision và preference odds-ratio objective trong một loss.

Trực giác:

1. Vẫn giữ năng lực language modeling hữu ích từ SFT.
2. Đồng thời đẩy xác suất chosen cao hơn rejected theo odds-ratio.

## 3. Vì sao dễ nhầm là “được hỗ trợ”

Nhiều hệ khác (TRL, LLaMA-Factory, Axolotl/TRL-wrapper) có ORPO trainer/mode.

Nhưng với `verl`, cần phân biệt rõ:

- “ecosystem có thể triển khai được”
- và “core repo có trainer ORPO sẵn để chạy trực tiếp”

## 4. Nếu muốn ORPO nhưng vẫn tận dụng `verl`

Lộ trình kỹ thuật:

1. Tạo trainer mới theo khung controller-workers của `verl`.
2. Thêm ORPO loss function registry + data loader cho preference pair.
3. Tái sử dụng hạ tầng rollout/distributed/checkpoint có sẵn.

## 5. Khuyến nghị thực tế

- Nhu cầu ORPO ngay: dùng TRL/LLaMA-Factory/Axolotl để có pipeline nhanh.
- Nhu cầu ORPO quy mô lớn với custom runtime: phát triển extension theo pattern `verl`.
