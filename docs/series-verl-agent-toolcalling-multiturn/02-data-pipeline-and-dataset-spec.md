# 02 - Data Pipeline & Dataset Spec

## 1. Nguyên tắc dữ liệu cho multi-turn tool-calling

1. Mỗi mẫu phải giữ đủ context nhiều lượt.
2. Tool call và tool result phải parse được máy.
3. Dữ liệu train/val/test tách theo task template để tránh leakage.

## 2. Schema tối thiểu đề xuất

```json
{
  "id": "sample_0001",
  "prompt": [
    {"role": "system", "content": "You are a tool-using assistant."},
    {"role": "user", "content": "Tìm giá BTC hôm nay rồi quy đổi sang VND."}
  ],
  "metadata": {
    "task_type": "toolcall_finance",
    "agent_name": "tool_agent_loop"
  }
}
```

Trong quá trình rollout, cần lưu thêm trace:

- `tool_calls`
- `tool_outputs`
- `assistant_turns`
- `final_answer`

## 3. Chuẩn bị dữ liệu cho `verl`

- Dạng parquet/json theo key `prompt` (mặc định của `RLHFDataset`).
- Bật `data.return_raw_chat=True` cho multi-turn.
- Với tool config, đặt `actor_rollout_ref.rollout.multi_turn.tool_config_path`.

## 4. Kiểm định dữ liệu trước train

Checklist bắt buộc:

1. Parse pass rate của message format >= 99%.
2. Tool argument schema validation pass >= 99%.
3. Không có mẫu vượt ngưỡng length quá mức cluster cho phép.
4. Có phân phối task types cân bằng tương đối.

## 5. Done Criteria

- Có script build dataset tái lập.
- Có report data quality (parse/length/schema).
- Có bộ val/test cố định.
