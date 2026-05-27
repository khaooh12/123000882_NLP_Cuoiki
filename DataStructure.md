# Cấu trúc Dữ liệu Dự án (Data Structure)

Tài liệu này mô tả chi tiết cấu trúc thư mục dữ liệu, thông số kỹ thuật của các tệp tin CSV và phân phối các chuyên mục (chủ đề) trong từng tập dữ liệu của dự án phân cụm văn bản tiếng Việt.

---

## 📂 Tổng quan Thư mục dữ liệu
Dữ liệu dự án được tổ chức thành hai thư mục chính:
- **`data/raw/`**: Chứa các tệp dữ liệu thô định dạng CSV được thu thập từ scraping VnExpress, tải từ Hugging Face Datasets hoặc sklearn.
- **`data/processed/`**: Chứa dữ liệu sau khi được làm sạch (cleaned), các vector embedding của mô hình SBERT và kết quả dự đoán trên tập kiểm thử.

---

## 1. Chi tiết dữ liệu thô (`data/raw/`)

### 1.1. `all_articles.csv`
- **Mô tả**: Tệp dữ liệu tổng hợp từ toàn bộ các nguồn tiếng Việt, sẵn sàng cho bước tiền xử lý (preprocessing).
- **Ngôn ngữ**: Tiếng Việt (`vi`)
- **Số dòng (bài viết)**: 67.711
- **Các cột (Columns)**: `url`, `title`, `description`, `category`, `publish_date`, `source`, `language`
- **Phân phối theo nguồn (source)**:
  - `vntc`: 48.916 bài (72.2%)
  - `wiki-qa`: 5.000 bài (7.4%)
  - `wiki-mini`: 5.000 bài (7.4%)
  - `wiki-dump`: 5.000 bài (7.4%)
  - `vnexpress`: 3.795 bài (5.6%)
- **Phân phối Chuyên mục (Category)**:
  - **Thời sự**: 11.837 bài (17.5%)
  - **Sức khỏe**: 10.293 bài (15.2%)
  - **Bách khoa**: 9.713 bài (14.3%) *(⚠️ Bách khoa là danh mục cho các bài Wikipedia không khớp đủ từ khoá — BERTopic sẽ tự phân cụm chúng)*
  - **Khoa học-CN**: 9.477 bài (14.0%)
  - **Du lịch**: 6.424 bài (9.5%)
  - **Kinh doanh**: 6.257 bài (9.2%)
  - **Thể thao**: 6.178 bài (9.1%)
  - **Pháp luật**: 5.768 bài (8.5%)
  - **Giáo dục**: 1.764 bài (2.6%)

### 1.2. `vnexpress_articles.csv`
- **Mô tả**: Dữ liệu thu thập từ VnExpress bằng công cụ cào web (web scraper).
- **Ngôn ngữ**: Tiếng Việt (`vi`)
- **Số dòng (bài viết)**: 3.795
- **Các cột (Columns)**: `url`, `title`, `description`, `category`, `publish_date` (không có cột `source`, `language`)
- **Phân phối Chuyên mục**:
  - Du lịch: 731 bài
  - Giáo dục: 724 bài
  - Kinh doanh: 717 bài
  - Thể thao: 715 bài
  - Pháp luật: 708 bài
  - Sức khỏe: 100 bài
  - Thời sự: 100 bài

### 1.3. `vntc_articles.csv`
- **Mô tả**: Dữ liệu phân loại văn bản tiếng Việt từ tập dữ liệu VNTC-10Topics trên Hugging Face.
- **Ngôn ngữ**: Tiếng Việt (`vi`)
- **Số dòng (bài viết)**: 48.916
- **Các cột (Columns)**: `url`, `title`, `description`, `category`, `publish_date`, `source`, `language`
- **Phân phối Chuyên mục**:
  - Sức khỏe: 10.000 bài
  - Thời sự: 10.000 bài
  - Khoa học-CN: 8.916 bài
  - Thể thao: 5.000 bài
  - Du lịch: 5.000 bài
  - Kinh doanh: 5.000 bài
  - Pháp luật: 5.000 bài

### 1.4. `wiki_qa_articles.csv`
- **Mô tả**: Bộ dữ liệu WikiQA được lấy từ Hugging Face, với tiêu đề lấy từ câu hỏi (question) và mô tả lấy từ ngữ cảnh (context).
- **Ngôn ngữ**: Tiếng Việt (`vi`)
- **Số dòng (bài viết)**: 5.000
- **Các cột (Columns)**: `url`, `title`, `description`, `category`, `publish_date`, `source`, `language`
- **Phân phối Chuyên mục**:
  - Bách khoa: 2.790 bài
  - Thời sự: 895 bài
  - Giáo dục: 416 bài
  - Du lịch: 379 bài
  - Kinh doanh: 260 bài
  - Thể thao: 123 bài
  - Sức khỏe: 57 bài
  - Khoa học-CN: 50 bài
  - Pháp luật: 30 bài

### 1.5. `wiki_mini_articles.csv`
- **Mô tả**: Dữ liệu lấy mẫu từ bộ dữ liệu `wiki-mini-corpus` trên Hugging Face.
- **Ngôn ngữ**: Tiếng Việt (`vi`)
- **Số dòng (bài viết)**: 5.000
- **Các cột (Columns)**: `url`, `title`, `description`, `category`, `publish_date`, `source`, `language`
- **Phân phối Chuyên mục**:
  - Bách khoa: 3.354 bài
  - Thời sự: 507 bài
  - Giáo dục: 353 bài
  - Thể thao: 245 bài
  - Kinh doanh: 181 bài
  - Du lịch: 148 bài
  - Khoa học-CN: 98 bài
  - Sức khỏe: 89 bài
  - Pháp luật: 25 bài

### 1.6. `wiki_dump_articles.csv`
- **Mô tả**: Dữ liệu lấy mẫu từ bản tóm tắt bài viết của dump tiếng Việt Wikipedia sạch (`wiki-dump-cleaned`).
- **Ngôn ngữ**: Tiếng Việt (`vi`)
- **Số dòng (bài viết)**: 5.000
- **Các cột (Columns)**: `url`, `title`, `description`, `category`, `publish_date`, `source`, `language`
- **Phân phối Chuyên mục**:
  - Bách khoa: 3.569 bài
  - Khoa học-CN: 413 bài
  - Thời sự: 335 bài
  - Giáo dục: 271 bài
  - Du lịch: 166 bài
  - Kinh doanh: 99 bài
  - Thể thao: 95 bài
  - Sức khỏe: 47 bài
  - Pháp luật: 5 bài

### 1.7. `newsgroups_articles.csv`
- **Mô tả**: Bộ dữ liệu 20 Newsgroups kinh điển lấy từ `sklearn.datasets.fetch_20newsgroups`. Dùng để kiểm tra độc lập hiệu năng của pipeline với tiếng Anh.
- **Ngôn ngữ**: Tiếng Anh (`en`) (Không gộp vào `all_articles.csv`)
- **Số dòng (bài viết)**: 3.000
- **Các cột (Columns)**: `url`, `title`, `description`, `category`, `publish_date`, `source`, `language`
- **Phân phối Chuyên mục**:
  - Khoa học-CN: 1.200 bài
  - Thời sự: 900 bài
  - Thể thao: 600 bài
  - Sức khỏe: 150 bài
  - Kinh doanh: 150 bài

### 1.8. `test_articles.csv`
- **Mô tả**: Dữ liệu cào độc lập từ báo VietnamNet, đóng vai trò là tập dữ liệu kiểm thử (test set) độc lập để đánh giá mô hình.
- **Ngôn ngữ**: Tiếng Việt (`vi`)
- **Số dòng (bài viết)**: 140
- **Các cột (Columns)**: `url`, `title`, `description`, `category`, `publish_date`, `source`, `language`
- **Phân phối Chuyên mục**: 20 bài viết cho mỗi chuyên mục trong 7 chuyên mục: Thời sự, Kinh doanh, Thể thao, Giáo dục, Sức khỏe, Pháp luật, Du lịch.

---

## 2. Chi tiết dữ liệu đã xử lý (`data/processed/`)

### 2.1. `cleaned_articles.csv`
- **Mô tả**: Dữ liệu sau khi thực hiện làm sạch văn bản (loại bỏ ký tự đặc biệt, chuẩn hóa tiếng Việt, tách từ và loại bỏ từ dừng).
- **Số dòng**: 67.711 (Tương ứng 1:1 với `all_articles.csv`)
- **Các cột**: `url`, `title`, `description`, `category`, `publish_date`, `source`, `language`, `cleaned_text` (cột mới chứa nội dung văn bản đã làm sạch).

### 2.2. `embeddings.npy`
- **Mô tả**: Ma trận vector embeddings được sinh ra bằng mô hình SBERT (`paraphrase-multilingual-MiniLM-L12-v2`). Được lưu dưới định dạng NumPy binary để tái sử dụng nhanh chóng.
- **Kích thước (Shape)**: `(67711, 384)`
- **Kiểu dữ liệu (Dtype)**: `float32`

### 2.3. `test_predictions.csv`
- **Mô tả**: Tệp kết quả dự đoán nhãn chủ đề từ mô hình LDA và BERTopic trên tập kiểm thử độc lập `test_articles.csv`.
- **Số dòng**: 140
- **Các cột**: `url`, `title`, `description`, `category`, `publish_date`, `source`, `language`, `combined`, `cleaned_text`, `lda_topic`, `bertopic_official`, `bertopic_fallback`.

---

## 📊 Bảng tổng hợp cấu trúc dữ liệu dự án

| Vị trí | Tên Tệp tin | Số dòng | Nguồn dữ liệu | Ngôn ngữ | Ghi chú |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `raw/` | **`all_articles.csv`** | **67.711** | Gộp các nguồn tiếng Việt | `vi` | Dữ liệu đầu vào chính cho tiền xử lý |
| `raw/` | `vnexpress_articles.csv` | 3.795 | Web scraping (VnExpress) | `vi` | Scraper nội bộ, có nhãn chuyên mục chuẩn |
| `raw/` | `vntc_articles.csv` | 48.916 | `kornwtp/VNTC-10Topics` | `vi` | Bộ dữ liệu lớn, làm phong phú từ vựng |
| `raw/` | `wiki_qa_articles.csv` | 5.000 | `vntc/WikiQA-84k` | `vi` | Lấy câu hỏi làm title, ngữ cảnh làm description |
| `raw/` | `wiki_mini_articles.csv` | 5.000 | `vntc/wiki-mini-corpus` | `vi` | Tập con lấy mẫu ngẫu nhiên từ Wikipedia |
| `raw/` | `wiki_dump_articles.csv` | 5.000 | `vntc/wiki-dump-cleaned` | `vi` | Bản làm sạch của Wikipedia dump tiếng Việt |
| `raw/` | `newsgroups_articles.csv` | 3.000 | `20 Newsgroups` (sklearn) | `en` | Dùng để test pipeline độc lập |
| `raw/` | `test_articles.csv` | 140 | Web scraping (VietnamNet) | `vi` | Tập kiểm thử độc lập (20 bài/chuyên mục × 7) |
| `processed/` | **`cleaned_articles.csv`** | **67.711** | Từ `all_articles.csv` | `vi` | Đã chuẩn hóa, tách từ bằng Underthesea |
| `processed/` | `embeddings.npy` | `(67711, 384)` | Từ SBERT embedding | - | Ma trận lưu trữ vector embeddings (float32) |
| `processed/` | `test_predictions.csv` | 140 | Kết quả dự đoán test set | `vi` | Chứa nhãn dự đoán từ LDA & BERTopic |

---
*Cập nhật lần cuối: 25/05/2026*