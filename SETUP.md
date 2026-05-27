# Hướng Dẫn Cài Đặt Môi Trường (SETUP)

Tài liệu này hướng dẫn cách thiết lập môi trường để chạy dự án **Phân Cụm Chủ Đề Văn Bản Tiếng Việt**.

## 1. Yêu Cầu Hệ Thống
- Python 3.9 trở lên (Khuyến nghị Python 3.10 hoặc 3.11).
- Hệ điều hành: Windows, macOS, hoặc Linux.
- Khuyến nghị có GPU CUDA nếu muốn tăng tốc quá trình tạo Embedding với `sentence-transformers` (không bắt buộc).

## 2. Các Bước Thiết Lập

### Bước 1: Tạo môi trường ảo (Virtual Environment)
Mở terminal/powershell tại thư mục gốc của dự án và chạy:

```bash
# Tạo môi trường ảo tên là venv
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows (PowerShell):
venv\Scripts\Activate.ps1

# Trên Windows (Command Prompt):
venv\Scripts\activate.bat

# Trên macOS/Linux:
source venv/bin/activate
```

### Bước 2: Cập nhật pip và setuptools
Đảm bảo pip và setuptools được cập nhật phiên bản mới nhất trước khi cài đặt các thư viện nặng:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Bước 3: Cài đặt các thư viện
Chạy lệnh cài đặt toàn bộ phụ thuộc từ `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 3. Hướng Dẫn Xử Lý Lỗi Cài Đặt (Đặc biệt trên Windows)

### Lỗi 1: Không cài được `hdbscan` (Lỗi thiếu C++ Build Tools)
Thư viện `hdbscan` cần trình biên dịch C++ để cài đặt trên Windows. Nếu gặp lỗi này, bạn có các cách xử lý sau:

1. **Cách khuyên dùng - Sử dụng cơ chế Dự phòng (Fallback):**
   Mã nguồn của chúng tôi đã được thiết kế thông minh: **Nếu hệ thống không import được `bertopic` hoặc `hdbscan`, hệ thống sẽ tự động chuyển sang mô hình dự phòng sử dụng TF-IDF + KMeans**. Bạn vẫn có thể chạy ứng dụng Streamlit và xem kết quả bình thường.
   
2. **Cài đặt Build Tools:**
   Tải và cài đặt [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), chọn gói **C++ build tools** trong trình cài đặt.
   
3. **Cài đặt qua conda (nếu sử dụng Anaconda):**
   ```bash
   conda install -c conda-forge hdbscan
   ```

### Lỗi 2: Lỗi cài đặt `pyLDAvis` với pandas mới
`pyLDAvis` thỉnh thoảng gặp lỗi tương thích với các phiên bản `pandas 2.x`.
Nếu gặp lỗi khi import `pyLDAvis` trong Streamlit:
- Giao diện Streamlit của chúng tôi sẽ tự động phát hiện lỗi và hiển thị cảnh báo thay vì làm treo ứng dụng.
- Bạn có thể khắc phục bằng cách hạ cấp pandas (nếu cần thiết):
  ```bash
  pip install pandas==1.5.3
  ```

---

## 4. Chạy Dự Án

Sau khi cài đặt xong, bạn có thể chạy các bước của pipeline như sau:

```bash
# 1. Cào dữ liệu
python src/scraper.py

# 2. Tiền xử lý dữ liệu
python src/preprocess.py

# 3. Tạo embeddings
python src/embedding.py

# 4. Huấn luyện mô hình
python src/topic_model.py

# 5. Chạy ứng dụng Streamlit
streamlit run app.py
```
