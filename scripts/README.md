# scripts/ — Script tiện ích (không thuộc pipeline chính)

Các script trong thư mục này được tạo trong quá trình phát triển để hỗ trợ kiểm tra, đánh giá, và thu thập dữ liệu bổ sung. Chúng **không** được gọi bởi `app.py` hay pipeline chính.

## Danh sách

| File | Mục đích | Cách chạy |
|------|----------|-----------|
| `evaluate_dataset.py` | Phân tích toàn diện dataset (thống kê, cân bằng, chất lượng) | `python scripts/evaluate_dataset.py` |
| `scrape_test_set.py` | Cào tập kiểm thử từ **VietnamNet** (nguồn độc lập với VnExpress) | `python scripts/scrape_test_set.py` |
| `test_model.py` | Đánh giá mô hình LDA + BERTopic trên tập kiểm thử VietnamNet | `python scripts/test_model.py` |

## Thứ tự chạy (nếu cần đánh giá độc lập)

```bash
# Bước 0: Đảm bảo pipeline chính đã chạy xong (scraper → preprocess → embedding → topic_model)

# Bước 1: Cào dữ liệu kiểm thử từ nguồn mới (VietnamNet)
python scripts/scrape_test_set.py

# Bước 2: Chạy đánh giá mô hình trên tập kiểm thử
python scripts/test_model.py

# Bước 3 (tuỳ chọn): Xem báo cáo thống kê dataset tổng hợp
python scripts/evaluate_dataset.py
```

> **Lưu ý:** Tất cả script phải được chạy từ **thư mục gốc dự án** (nơi chứa `app.py`), không phải từ bên trong `scripts/`.
