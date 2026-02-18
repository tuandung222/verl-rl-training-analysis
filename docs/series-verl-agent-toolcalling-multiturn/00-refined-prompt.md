# Refined Prompt (Phiên bản chỉn chu)

Bạn hãy viết một series tài liệu thực chiến, end-to-end, hướng dẫn dùng `verl` để huấn luyện LLM Agent cho bài toán **tool-calling multi-turn**.

## Mục tiêu

- Giúp người mới có thể đi từ zero đến chạy được một pipeline huấn luyện hoàn chỉnh.
- Làm rõ từng lớp: dữ liệu, reward, config, rollout, training loop, đánh giá, debug, tối ưu.
- Ưu tiên cách làm có thể tái lập trên dự án thực tế.

## Phạm vi bắt buộc

1. Thiết kế bài toán và yêu cầu Agent tool-calling multi-turn.
2. Chuẩn hóa dữ liệu:
   - schema hội thoại đa lượt,
   - schema tool + tool response,
   - format parquet/json cho `verl`,
   - kiểm định dữ liệu trước train.
3. Thiết kế reward cho tool-calling:
   - outcome reward,
   - format/parse reward,
   - efficiency/safety penalties,
   - reward decomposition và weighting.
4. Thiết kế training config `verl`:
   - `data`, `actor_rollout_ref`, `rollout.multi_turn`, `algorithm`, `trainer`,
   - các lỗi cấu hình thường gặp và cách tránh.
5. Chạy training end-to-end với ví dụ lệnh thực tế.
6. Đánh giá + debug:
   - metric online/offline,
   - rollout trace,
   - phân tích failure modes.
7. Tối ưu hiệu năng:
   - memory/throughput,
   - rollout bottleneck,
   - scale từ 1 node lên multi-node.
8. Checklist triển khai production.

## Yêu cầu chất lượng

- Viết theo phong cách thực dụng, có command/config cụ thể.
- Mỗi phần có checklist “Done Criteria”.
- Có sơ đồ minh họa luồng dữ liệu và luồng huấn luyện.
- Nêu rõ assumption, trade-off, rủi ro.

## Đầu ra

- Một series nhiều bài, có thứ tự học rõ ràng.
- Có mục lục tổng và roadmap thực hành 2-4 tuần.
