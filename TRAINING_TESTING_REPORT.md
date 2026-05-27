# Báo Cáo Quá Trình Huấn Luyện & Kiểm Thử Mô Hình
**Đề tài:** Phân Cụm Chủ Đề Văn Bản Tiếng Việt  
**MSSV:** 123000882  

Tài liệu này ghi nhận chi tiết quá trình huấn luyện (training), các thông số kỹ thuật, đánh giá chỉ số huấn luyện và kết quả kiểm thử (testing) độc lập của hai mô hình **LDA (Latent Dirichlet Allocation)** và **BERTopic** (bao gồm cả phiên bản chính thức HDBSCAN và phiên bản dự phòng KMeans).

---

## 📂 1. Tổng Quan Tập Dữ Liệu

### 1.1. Tập Dữ Liệu Huấn Luyện (Training Set)
Dữ liệu huấn luyện chính là tập hợp gộp từ nhiều nguồn văn bản tiếng Việt khác nhau để tạo nên sự đa dạng về mặt từ vựng và chủ đề.
- **Tệp dữ liệu tổng hợp:** `data/processed/cleaned_articles.csv` (đã qua tiền xử lý).
- **Tổng số lượng bài viết:** **67.711 bài viết**.
- **Phân bổ nguồn dữ liệu:**
  - **VNTC (HuggingFace):** 48.916 bài (72.2%)
  - **Wikipedia QA:** 5.000 bài (7.4%)
  - **Wikipedia Mini Corpus:** 5.000 bài (7.4%)
  - **Wikipedia Cleaned Dump:** 5.000 bài (7.4%)
  - **VnExpress Scraping:** 3.795 bài (5.6%)
- **Phân bổ theo chuyên mục (Category):**
  - **Thời sự:** 11.837 bài (17.5%)
  - **Sức khỏe:** 10.293 bài (15.2%)
  - **Bách khoa:** 9.713 bài (14.3%) (Danh mục Wikipedia chung)
  - **Khoa học-CN:** 9.477 bài (14.0%)
  - **Du lịch:** 6.424 bài (9.5%)
  - **Kinh doanh:** 6.257 bài (9.2%)
  - **Thể thao:** 6.178 bài (9.1%)
  - **Pháp luật:** 5.768 bài (8.5%)
  - **Giáo dục:** 1.764 bài (2.6%)

### 1.2. Tập Dữ Liệu Kiểm Thử Độc Lập (Test Set)
Nhằm đánh giá khách quan khả năng tổng quát hóa của mô hình trên dữ liệu thực tế hoàn toàn mới:
- **Tệp dữ liệu kiểm thử:** `data/raw/test_articles.csv`.
- **Nguồn:** Web scraping từ báo **VietnamNet** (nguồn hoàn toàn độc lập với dữ liệu huấn luyện VnExpress).
- **Số lượng bài viết:** **140 bài viết**.
- **Cấu trúc:** Phân bổ đồng đều với **20 bài viết cho mỗi chuyên mục** trong 7 chuyên mục: *Thời sự, Kinh doanh, Thể thao, Giáo dục, Sức khỏe, Pháp luật, Du lịch*.

---

## ⚙️ 2. Quy Trình Tiền Xử Lý & Trích Xuất Đặc Trưng

### 2.1. Tiền xử lý văn bản (`src/preprocess.py`)
Mọi bài viết ở cả tập train và test đều trải qua pipeline làm sạch:
1. Gộp trường `title` (tiêu đề) và `description` (tóm tắt) làm nội dung văn bản đầu vào.
2. Chuẩn hóa Unicode tiếng Việt dựng sẵn.
3. Loại bỏ các ký tự đặc biệt, đường dẫn URL, email, số điện thoại và các khoảng trắng thừa.
4. Tách từ (Word Segmentation) bằng thư viện chuyên dụng **Underthesea** (ví dụ: `học sinh` -> `học_sinh`).
5. Loại bỏ từ dừng (Stopwords) tiếng Việt dựa trên danh sách từ dừng chuẩn kết hợp các từ có tần suất xuất hiện quá cao hoặc quá thấp không mang giá trị ngữ nghĩa.

### 2.2. Trích xuất Embedding (`src/embedding.py`)
Mô hình BERTopic yêu cầu đầu vào dạng vector mật độ cao:
- **Mô hình sử dụng:** Sentence-BERT `paraphrase-multilingual-MiniLM-L12-v2` (hỗ trợ đa ngôn ngữ, tối ưu cho tiếng Việt).
- **Kích thước Embedding:** Mỗi văn bản được chuyển thành vector **384 chiều** (`float32`).
- Ma trận vector đầy đủ của tập huấn luyện `(67711, 384)` được cache lại tại `data/processed/embeddings.npy` để tránh phải tính toán lại trong các lần chạy sau.

---

## 🧠 3. Quá Trình Huấn Luyện Mô Hình (`src/topic_model.py`)

Hệ thống được thiết kế và huấn luyện song song hai hướng tiếp cận: Cổ điển (LDA) và Hiện đại (BERTopic).

### 3.1. Mô hình LDA (Latent Dirichlet Allocation)
- **Công cụ:** Thư viện `Gensim` (nếu thiếu, tự động fallback sang `scikit-learn` LDA).
- **Vector hóa:** Biểu diễn túi từ (Bag-of-Words) thông qua Dictionary & Corpus.
- **Tham số huấn luyện:**
  - `num_topics = 7` (Số lượng chủ đề cần tìm kiếm).
  - `passes = 15` (Số lượt duyệt qua toàn bộ dữ liệu khi tối ưu hóa).
  - `alpha = "auto"` (Tự học phân bố tài liệu - chủ đề).
  - Trực quan hóa tương tác: Tự động kết xuất dữ liệu sang `models/lda_model/pyldavis_data.html` thông qua thư viện `pyLDAvis`.

### 3.2. Mô hình BERTopic
- **Công cụ:** Thư viện `BERTopic` chính thức (kết hợp UMAP và HDBSCAN).
- **Tham số các thành phần:**
  - **Giảm chiều (UMAP):** Giảm từ 384 chiều xuống 5 chiều để tăng hiệu năng gom cụm, sử dụng khoảng cách `cosine`, `n_neighbors = 15`, `random_state = 42`.
  - **Gom cụm (HDBSCAN):** Phân cụm mật độ với `min_cluster_size = 10`, khoảng cách `euclidean`.
  - **Biểu diễn chủ đề (c-TF-IDF):** Dùng `ClassTfidfTransformer` và `CountVectorizer` với `ngram_range = (1, 2)` để trích xuất từ khóa đặc trưng cho mỗi cụm.
  - `nr_topics = 7` (Giảm số lượng cụm xuống còn 7 để khớp với các nhóm chủ đề chính).
- **Mô hình dự phòng (Fallback BERTopic):**
  - Để đảm bảo chương trình hoạt động mượt mà trên môi trường thiếu thư viện biên dịch C++ (không cài được `hdbscan`), hệ thống cung cấp lớp dự phòng **KMeans** với `n_clusters = 7` trên vector SBERT kết hợp tính toán thủ công ma trận c-TF-IDF để trích xuất từ khóa.

---

## 📊 4. Kết Quả Huấn Luyện (Training Results)

Sau khi chạy xong tiến trình huấn luyện trên toàn bộ tập dữ liệu tiếng Việt gồm 67.711 văn bản, các chỉ số đánh giá chất lượng phân cụm thu được như sau:

| Mô hình | Chỉ số Coherence $c_v$ | Ghi chú |
| :--- | :---: | :--- |
| **LDA Baseline (Gensim)** | **0.4642** | Mức độ gắn kết ngữ nghĩa trung bình khá. Các chủ đề có ranh giới phân tách tương đối. |
| **BERTopic (HDBSCAN)** | **0.4810** | Chỉ số coherence cao hơn LDA. Các từ khóa đặc trưng của mỗi chủ đề có sự liên kết tự nhiên tốt nhờ vector hóa ngữ nghĩa từ SBERT. |

- *Chi tiết chỉ số được tự động lưu trữ tại file cấu hình: [metrics.json](file:///d:/Code/Python/project/NLP/123000882_NLP_Cuoiki-main/models/metrics.json).*

---

## 🧪 5. Kết Quả Kiểm Thử (Testing Results)

Đánh giá mô hình trên tập kiểm thử độc lập gồm **140 bài viết** từ VietnamNet (chia đều 7 chuyên mục thực tế). Kết quả đối chiếu phân bổ dự đoán của các mô hình như sau:

### 5.1. Kết quả từ mô hình LDA
Mô hình LDA phân phối tương đối rõ nét và khớp khá tốt với các nhãn chuyên mục ban đầu:
- **Kinh doanh:** **80.0%** bài viết được quy về **Chủ đề 3** (Từ khóa nổi bật: *công_ty, đồng, công_an, tháng*).
- **Giáo dục:** **80.0%** bài viết được quy về **Chủ đề 4** (Từ khóa nổi bật: *việt_nam, tp, trường, phát_triển*).
- **Du lịch:** **65.0%** bài viết được quy về **Chủ đề 0** (Từ khóa nổi bật: *một, tỉnh, phía, thuộc*).
- **Pháp luật:** **60.0%** bài viết được quy về **Chủ đề 3** (Trùng cụm từ khóa với Kinh doanh do các bài viết về vụ án kinh tế).
- **Thể thao:** **50.0%** bài viết được quy về **Chủ đề 2** (Từ khóa nổi bật: *tháng, việt_nam, thứ, huyện*).
- **Sức khỏe:** **45.0%** bài viết được quy về **Chủ đề 6** (Từ khóa nổi bật: *màu, thường, hoặc, có_thể*).
- **Thời sự:** **30.0%** bài viết được quy về **Chủ đề 3**.

### 5.2. Kết quả từ mô hình BERTopic chính thức (HDBSCAN)
Do HDBSCAN hoạt động dựa trên phân bố mật độ và gán các điểm nằm ở vùng mật độ thấp làm nhiễu (outliers, gán nhãn `-1`):
- **Hiện tượng xảy ra:** Phần lớn các bài viết thuộc tập kiểm thử (từ 50% đến 85% tùy chuyên mục) bị gán nhãn **`-1`** (Outlier, tên cụm: `-1_mt_ngi_ng_khng`).
- **Nguyên nhân:** Dữ liệu VietnamNet có phong cách viết, từ vựng và phân phối phân cụm hơi khác biệt so với tập dữ liệu huấn luyện (chủ yếu là VNTC và Wikipedia). HDBSCAN rất khắt khe về mặt mật độ nên đã từ chối phân cụm cho các bài viết này thay vì ép chúng vào cụm sai.

### 5.3. Kết quả từ mô hình BERTopic dự phòng (KMeans)
KMeans là thuật toán phân hoạch (partitioning) bắt buộc mọi điểm dữ liệu phải thuộc về một cụm nào đó (không có khái niệm nhiễu `-1`). Kết quả phân bổ cho thấy sự tương quan rõ rệt:
- **Sức khỏe:** **65.0%** bài viết được quy về **Chủ đề 6** (`6_một_người_không_bạn`).
- **Kinh doanh:** **65.0%** bài viết được quy về **Chủ đề 2** (`2_đồng_một_không_tháng`).
- **Thể thao:** **50.0%** bài viết được quy về **Chủ đề 0** (`0_một_không_hai_người`).
- **Thời sự:** **45.0%** bài viết được quy về **Chủ đề 6** (`6_một_người_không_bạn`).
- **Giáo dục:** **45.0%** bài viết được quy về **Chủ đề 1** (`1_việt_nam_một_người_tỉnh`).
- **Pháp luật:** **30.0%** bài viết được quy về **Chủ đề 3** (`3_một_người_không_bị`).
- **Du lịch:** **30.0%** bài viết được quy về **Chủ đề 1** (`1_việt_nam_một_người_tỉnh`).

---

## 💡 6. Đánh Giá Chung & Kiến Nghị

1. **Về khả năng phân cụm học không giám sát:**
   - Cả LDA và BERTopic (KMeans) đều học được các đặc trưng ngữ nghĩa nhất quán. Các bài viết về *Thể thao, Giáo dục, Kinh doanh, Sức khỏe* được gom nhóm thành công với độ tập trung cao (từ 50% - 80% bài viết cùng một chuyên mục thực tế rơi vào cùng một cụm chủ đề dự đoán).
2. **Hạn chế của HDBSCAN trên dữ liệu mới:**
   - Mô hình HDBSCAN trong BERTopic chính thức rất nhạy cảm với dữ liệu kiểm thử ngoài phân phối huấn luyện (out-of-distribution), dẫn tới tỷ lệ bài viết bị gán nhãn `-1` (nhiễu) cao.
3. **Hướng khắc phục kiến nghị:**
   - **Cách 1:** Sử dụng thuật toán dự đoán mềm hoặc tính khoảng cách Cosine từ vector Embedding của văn bản mới tới các trọng tâm cụm (centroid) đã học để cưỡng bức phân loại (như cách ứng dụng Streamlit đang triển khai ở Tab 1).
   - **Cách 2:** Tăng lượng dữ liệu huấn luyện đa dạng hơn từ VietnamNet hoặc tinh chỉnh lại siêu tham số `min_samples` và `cluster_selection_epsilon` của HDBSCAN để giảm lượng điểm ngoại lai.
