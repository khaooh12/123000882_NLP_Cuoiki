# Phân Cụm Chủ Đề Văn Bản Tiếng Việt

> **MSSV:** 123000882  
> **Môn học:** Xử Lý Ngôn Ngữ Tự Nhiên (NLP) — Cuối Khoá  
> **Đề tài:** Phân cụm chủ đề văn bản tiếng Việt (Vietnamese Topic Modeling & Clustering)

---

## Mô Tả Đề Tài

Cho một tập hợp văn bản tiếng Việt, hệ thống tự động:

1. Tiền xử lý, tách từ bằng `underthesea`
2. Biểu diễn văn bản bằng TF-IDF
3. Phân cụm không giám sát bằng KMeans
4. Đánh giá bằng chỉ số nội sinh (Silhouette, Coherence c_v) và ngoại sinh (NMI, ARI, Purity)
5. Trực quan hoá phân bố cụm và từ khoá đặc trưng qua giao diện Streamlit

```
Văn bản thô  ──►  Tiền xử lý  ──►  TF-IDF Vector  ──►  KMeans  ──►  Chủ đề + Visualize
                 (underthesea)    (5.000 đặc trưng)   (k = 3–12)
```

### Ví dụ kết quả

| Văn bản | Chủ đề phát hiện |
|---------|-----------------|
| *"VNIndex tăng 15 điểm, nhà đầu tư mua mạnh cổ phiếu ngân hàng"* | Kinh doanh |
| *"Đội tuyển Việt Nam thắng 2-0 trước Thái Lan tại AFF Cup"* | Thể thao |
| *"Bộ GD-ĐT công bố điểm chuẩn đại học năm 2025"* | Giáo dục |

---

## Cấu Trúc Dự Án

```
123000882_NLP_Cuoiki/
│
├── app.py                          # Streamlit demo (entry point)
├── requirements.txt
├── README.md
│
├── src/
│   ├── preprocess.py               # Tiền xử lý, tách từ tiếng Việt (underthesea)
│   ├── topic_model.py              # FallbackTopicModel (KMeans + c-TF-IDF), LDA
│   └── visualize.py                # Scatter plot 2D, word cloud
│
├── scripts/
│   ├── run_tfidf_kmeans.py         # Thực nghiệm TF-IDF + KMeans (k=3..10), lưu kết quả
│   └── evaluate_external.py        # Đánh giá ngoại sinh: NMI, ARI, Purity, confusion matrix
│
├── data/
│   ├── raw/                        # CSV từ nhiều nguồn báo (all_articles.csv + từng nguồn)
│   └── processed/
│       └── cleaned_articles.csv    # Dữ liệu sau tiền xử lý (82.495 bài, 8 chủ đề)
│
└── results/
    ├── evaluation_report.md        # Báo cáo đánh giá đầy đủ (nội sinh + ngoại sinh)
    ├── tfidf_kmeans_results.json   # Silhouette, Coherence theo từng k
    ├── tfidf_kmeans_results.csv
    ├── external_eval_results.json  # NMI, ARI, Purity, V-measure theo từng k
    ├── external_eval_results.csv
    ├── confusion_matrix_k7.csv     # Ma trận nhầm lẫn cụm vs nhãn thực (k=7)
    ├── confusion_matrix_k8.csv
    └── confusion_matrix_k9.csv
```

---

## Dữ Liệu

| Thuộc tính | Giá trị |
|-----------|---------|
| **Tổng số bài** | 82.495 |
| **Nguồn** | VnExpress, DanTri, Tuổi Trẻ, Thanh Niên, Nhân Dân, Tiền Phong, VietnamNet, ZNews, VNTC |
| **8 chủ đề (ground truth)** | Thời sự, Sức khỏe, Khoa học-CN, Kinh doanh, Thể thao, Du lịch, Pháp luật, Giáo dục |
| **Đặc trưng** | Tiêu đề + Mô tả, sau khi tách từ và loại stopword |

---

## Demo Streamlit

### Chạy

```bash
streamlit run app.py
```

Truy cập `http://localhost:8501`.

### Giao diện

**Sidebar:** Chọn số chủ đề (3–12, mặc định 7).

**Hai tab nhập liệu:**

| Tab | Mô tả |
|-----|-------|
| **Upload File** | CSV (cần cột `text` / `title` / `description` / `content` / `cleaned_text`) hoặc TXT (mỗi dòng 1 văn bản). Tối đa 2.000 văn bản. |
| **Nhập Văn Bản** | Dán trực tiếp, mỗi dòng = 1 câu/đoạn. |

**Kết quả hiển thị:**

| Thành phần | Mô tả |
|-----------|-------|
| Danh sách chủ đề | Bảng tên chủ đề, số bài, từ khoá đại diện |
| Scatter Plot 2D | Mỗi điểm là 1 văn bản, màu theo cụm (t-SNE / PCA fallback) |
| Từ khoá đặc trưng | Badge từ khoá + Word cloud cho từng chủ đề |
| Văn bản tô màu | Mỗi đoạn được tô nền theo màu chủ đề |
| Download CSV | Xuất kết quả phân cụm |

---

## Phương Pháp

### Pipeline

| Bước | Công cụ | Chi tiết |
|------|---------|---------|
| Tiền xử lý | `underthesea` | Tách từ, loại stopwords tiếng Việt, chuẩn hoá Unicode NFC |
| Vector hoá | `TfidfVectorizer` | 5.000 đặc trưng, `min_df=3`, `max_df=0.85` |
| Phân cụm | `KMeans` | Không giám sát, số cụm tuỳ chọn (3–12) |
| Từ khoá cụm | c-TF-IDF thủ công | Top-10 từ đặc trưng nhất mỗi cụm |
| Giảm chiều | `t-SNE` → `PCA` (fallback) | Hiển thị phân bố 2D trong Streamlit |

### Đặt tên chủ đề tự động

Hệ thống so khớp từ khoá của mỗi cụm với từ điển `TOPIC_SIGNATURES` gồm 8 nhóm chủ đề (Thể thao, Kinh doanh, Sức khỏe, Thời sự, Giáo dục, Du lịch, Pháp luật, Khoa học-CN). Nếu không khớp → fallback về top-3 từ khoá.

---

## Kết Quả Thực Nghiệm

Thực nghiệm với k = 3 → 10 trên 82.495 bài (TF-IDF 5.000 đặc trưng):

### Chỉ số nội sinh

| k | Silhouette | Coherence c_v |
|---|:----------:|:-------------:|
| 3 | 0.0054 | 0.4900 |
| 5 | 0.0089 | 0.6556 |
| 7 | 0.0116 | 0.7011 |
| **9** | 0.0102 | **0.7130** |
| **10** | **0.0124** | 0.7054 |

### Chỉ số ngoại sinh (so với 8 nhãn ground truth)

| k | NMI | ARI | Purity |
|---|:---:|:---:|:------:|
| 7 | 0.3577 | 0.2059 | 0.5233 |
| **8** | 0.3649 | **0.2391** | **0.5499** |
| **9** | **0.3696** | 0.1828 | 0.5015 |

**Nhận xét nổi bật:**
- **Thể thao** được tách tự nhiên với độ thuần **98%+** ở mọi k ≥ 7 — từ vựng ngành rất đặc thù.
- **Khoa học-CN** đạt **89.9%** độ thuần khi k=8.
- **Thời sự** khó tách nhất — luôn là cụm lớn nhất (30–45%), từ vựng chồng lấn nhiều chủ đề.
- **k=8** khớp nhãn thực tốt nhất (ARI=0.2391, Purity=0.5499); **k=9** có Coherence cao nhất (0.7130).

Xem chi tiết tại [results/evaluation_report.md](results/evaluation_report.md).

---

## Cài Đặt & Chạy

### Yêu cầu hệ thống

- Python 3.10+
- RAM 4 GB+
- Không cần GPU

### Cài đặt

```bash
git clone https://github.com/khaooh12/123000882_NLP_Cuoiki
cd 123000882_NLP_Cuoiki

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Tiền xử lý dữ liệu

```bash
# Xử lý all_articles.csv (mặc định)
python src/preprocess.py

# Tuỳ chọn: chỉ lấy bài tiếng Việt
python src/preprocess.py --lang vi

# Chỉ định file input/output khác
python src/preprocess.py --input data/raw/vnexpress_articles.csv --output data/processed/vnexpress_cleaned.csv
```

### Chạy thực nghiệm & đánh giá

```bash
# Thực nghiệm TF-IDF + KMeans k=3..10, tính Silhouette & Coherence
python scripts/run_tfidf_kmeans.py

# Đánh giá ngoại sinh (NMI, ARI, Purity, confusion matrix) — cần cột category
python scripts/evaluate_external.py
```

Kết quả lưu vào thư mục `results/`.

### Chạy demo Streamlit

```bash
streamlit run app.py
```

---

## Stack Công Nghệ

| Thành phần | Thư viện |
|-----------|---------|
| Tiền xử lý NLP | `underthesea` |
| Machine Learning | `scikit-learn` |
| Topic Coherence | `gensim` |
| Visualization | `plotly`, `wordcloud`, `matplotlib` |
| Web App | `streamlit` |

---

## Tài Nguyên

- [underthesea](https://github.com/undertheseanlp/underthesea) — NLP tiếng Việt
- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Gensim CoherenceModel](https://radimrehurek.com/gensim/models/coherencemodel.html)
- [Streamlit](https://streamlit.io)
