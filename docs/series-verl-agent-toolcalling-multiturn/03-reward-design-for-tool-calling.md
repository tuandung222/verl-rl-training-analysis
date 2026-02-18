# 03 - Reward Design for Tool-Calling

## 1. Vì sao reward decomposition là bắt buộc

Outcome đúng chưa đủ. Agent có thể:

- gọi tool sai format,
- gọi thừa nhiều lần,
- trả lời đúng do may mắn ngắn hạn.

Cần reward nhiều thành phần.

## 2. Reward template gợi ý

`R_total = w1*R_task + w2*R_tool_format + w3*R_tool_efficiency + w4*R_safety`

Ví dụ:

- `R_task`: 1 nếu đúng kết quả, 0 nếu sai.
- `R_tool_format`: +0.2 nếu tool call parseable + schema-valid.
- `R_tool_efficiency`: -0.05 mỗi call dư sau ngưỡng kỳ vọng.
- `R_safety`: -1 nếu vi phạm policy/tool misuse.

## 3. Tổ chức custom reward function trong `verl`

Cấu hình:

- `reward.custom_reward_function.path=/abs/path/to/reward.py`
- `reward.custom_reward_function.name=compute_score`

Function nên trả token-level hoặc sequence-level score nhất quán.

## 4. Anti-pattern cần tránh

1. Reward sparse quá mức -> learning chậm.
2. Weight lệch lớn khiến model optimize metric phụ.
3. Không log từng term reward -> khó debug reward hacking.

## 5. Done Criteria

- Có công thức reward rõ và rationale.
- Có dashboard từng term reward.
- Có stress-test reward bằng bộ trace “xấu” cố ý.
