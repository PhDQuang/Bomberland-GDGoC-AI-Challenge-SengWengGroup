# My Custom Agent

Đây là thư mục chứa Agent của tôi để tham gia thi đấu.

## Cấu trúc:
- `agent.py`: File code chính bắt buộc (chứa class `Agent`).
- `utils.py`: Chứa các hàm tính toán phụ trợ.

## Cách chạy test:
Mở terminal và chạy lệnh sau từ thư mục gốc của dự án:
```bash
python -m scripts.participant.run_local_match --agent_paths agent/my_agent/ None None None --visualize true
```
