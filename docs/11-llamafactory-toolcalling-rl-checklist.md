# Lưu ý khi setup LLaMA-Factory để train RL tool-calling

Nguồn tham chiếu chính:

- Arguments: https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html
- Data preparation: https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html

## 1. Chọn stage đúng với objective

Theo docs arguments, `stage` có các giá trị gồm `ppo`, `dpo`, `kto` (và các stage khác).

Chọn sai stage là lỗi phổ biến nhất.

## 2. Dữ liệu tool-calling phải đúng schema

`tool-calling` thường nên theo ShareGPT/OpenAI-like messages với role tường minh.

Điểm quan trọng:

1. Mọi turns phải tuần tự đúng (user/assistant/function...).
2. Tool call và tool result phải có format thống nhất.
3. Nếu dùng preference training (DPO/ORPO), chosen/rejected phải cùng prompt context.

## 3. Các RLHF arguments cần khóa sớm

Nhóm preference:

- `pref_beta`
- `pref_loss` (`sigmoid`, `hinge`, `ipo`, `kto_pair`, `orpo`, `simpo`)
- `dpo_label_smoothing`
- `kto_chosen_weight`, `kto_rejected_weight`

Nhóm PPO:

- `ppo_epochs`
- `ppo_target`
- `ppo_score_norm`
- `ppo_whiten_rewards`
- `reward_model*`, `ref_model*`

## 4. Checklist data quality cho tool-calling

1. Đảm bảo tool schema xuất hiện đúng nơi model kỳ vọng.
2. Loại mẫu có vòng lặp tool vô hạn.
3. Loại samples tool output quá dài không cần thiết.
4. Gắn cờ samples fail parse để tách phân tích.

## 5. Reward design cho tool-calling

Một reward đơn lẻ thường không đủ.

Nên decomposition:

- `R_task`: đúng bài toán.
- `R_tool_format`: call hợp lệ.
- `R_tool_efficiency`: ít call thừa.
- `R_safety`: không vi phạm policy.

## 6. Hyperparameter guardrails

1. `beta` quá lớn dễ kéo model sát reference và chậm học.
2. sequence cutoff thấp sẽ cắt mất context tool -> học sai hành vi.
3. batch quá nhỏ làm preference signal nhiễu.

## 7. Observability tối thiểu

1. Log parse rate của tool call.
2. Log completion rate theo số turn.
3. Log reward từng thành phần.
4. Sample traces định kỳ để review bằng mắt.

## 8. Khi nào chọn LLaMA-Factory vs `verl` cho tool-calling RL

- Chọn LLaMA-Factory nếu ưu tiên trải nghiệm nhanh, tích hợp stage, dễ vận hành với team nhỏ.
- Chọn `verl` nếu cần runtime RL quy mô lớn, multi-turn async rollout và custom dataflow sâu.
