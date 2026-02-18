# 06 - Evaluation, Debugging, and Iteration

## 1. Bộ metric bắt buộc

1. `Task Success Rate`
2. `Tool Parse Success Rate`
3. `Average Tool Calls / Episode`
4. `Invalid Tool Call Rate`
5. `Latency per Episode`

## 2. Failure modes thường gặp

1. Tool hallucination (gọi tool không tồn tại).
2. Loop vô hạn giữa suy nghĩ và gọi tool.
3. Trả lời đúng format nhưng sai nội dung.
4. Reward hacking (điểm cao nhưng UX tệ).

## 3. Quy trình debug

1. Lấy mẫu fail theo từng task type.
2. Soi trace đa lượt và xác định turn lỗi đầu tiên.
3. Gán lỗi vào bucket: data / reward / config / tool runtime.
4. Sửa 1 biến mỗi vòng và chạy lại A/B.

## 4. Done Criteria

- Có failure taxonomy.
- Có playbook xử lý top 5 lỗi lặp lại.
- Có loop cải tiến theo tuần với metric rõ.
