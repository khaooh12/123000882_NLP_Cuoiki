# Phân Cụm Chủ Đề Văn Bản Tiếng Việt

> **MSSV:** 123000882  
> **Môn học:** Xử Lý Ngôn Ngữ Tự Nhiên (NLP) — Cuối Khoá  
> **Đề tài:** Phân cụm chủ đề văn bản tiếng Việt (Vietnamese Topic Modeling & Clustering)

---

## Mô Tả Đề Tài

Cho một tập hợp văn bản tiếng Việt **không có nhãn**, hệ thống tự động:

1. Khám phá các chủ đề tiềm ẩn (unsupervised topic modeling)
2. Gán nhãn chủ đề cho từng văn bản
3. Trực quan hoá phân bố cụm và từ khoá đặc trưng

```
Văn bản thô  ──►  Tiền xử lý  ──►  SBERT Embedding  ──►  KMeans/BERTopic  ──►  Chủ đề + Visualize
```

### Ví dụ

| Văn bản | Chủ đề phát hiện |
|---------|-----------------|
| *"VNIndex tăng 15 điểm, nhà đầu tư mua mạnh cổ phiếu ngân hàng"* | Tài chính — Chứng khoán |
| *"Đội tuyển Việt Nam thắng 2-0 trước Thái Lan tại AFF Cup"* | Thể thao |
| *"Bộ GD-ĐT công bố điểm chuẩn đại học năm 2025"* | Giáo dục |

---

## Cấu Trúc Dự Án

```
123000882_NLP_Cuoiki/
│
├── app.py                      # ★ Streamlit demo (entry point)
├── requirements.txt
├── README.md
│
├── src/                        # Module pipeline cốt lõi
│   ├── preprocess.py           # Tiền xử lý, chuẩn hoá tiếng Việt
│   ├── embedding.py            # Tạo sentence embeddings (SBERT)
│   ├── topic_model.py          # KMeans Fallback + BERTopic + LDA
│   ├── visualize.py            # Scatter plot, word cloud, bar chart
│   ├── data_loader.py          # Tải & gộp dữ liệu đa nguồn
│   └── scraper.py              # Thu thập bài báo từ VnExpress
│
├── scripts/                    # Script tiện ích (chạy một lần)
│   ├── scrape_test_set.py      # Cào tập kiểm thử từ VietnamNet
│   ├── test_model.py           # Đánh giá mô hình trên tập kiểm thử
│   └── evaluate_dataset.py     # Thống kê toàn bộ dataset
│
├── data/
│   ├── raw/                    # Dữ liệu thô — bị gitignore (tái tạo qua scraper)
│   └── processed/              # Embeddings & cleaned CSV — bị gitignore
│
└── models/
    ├── bertopic_model/         # Model đã train — bị gitignore
    └── lda_model/              # Model LDA — bị gitignore
```

---

## Demo Streamlit

Giao diện demo thực hiện luồng:

**Upload CSV/TXT → Phân cụm KMeans → Hiển thị kết quả**

| Thành phần | Mô tả |
|-----------|-------|
| **Danh sách chủ đề** | Bảng tên chủ đề, số bài, từ khoá đại diện |
| **Scatter Plot 2D** | Mỗi điểm là 1 văn bản, màu theo cụm (UMAP/PCA giảm chiều) |
| **Từ khoá đặc trưng** | Badge từ khoá + Word cloud cho từng chủ đề |
| **Download CSV** | Xuất kết quả phân cụm |

---

## Phương Pháp

### Pipeline chính

| Bước | Công cụ | Vai trò |
|------|---------|---------|
| Tiền xử lý | `underthesea` | Tách từ, loại stopwords, chuẩn hoá Unicode |
| Embedding | `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2` | Vector 384-dim đa ngôn ngữ |
| Clustering | `KMeans` (fallback) / `HDBSCAN` (BERTopic) | Phân nhóm không giám sát |
| Từ khoá | `c-TF-IDF` | Trích xuất từ đặc trưng mỗi cụm |
| Giảm chiều | `UMAP` / `PCA` | Hiển thị 2D |

### Hai hướng so sánh

- **BERTopic** (chủ đạo): SBERT → UMAP → HDBSCAN → c-TF-IDF
- **LDA** (baseline): TF-IDF → Gensim LDA → pyLDAvis

---

## Dữ Liệu

| Nguồn | Phương pháp | Số lượng |
|-------|------------|---------|
| VnExpress | Web scraping (`requests` + `BeautifulSoup`) | ~10.000 bài |
| VNTC | HuggingFace Datasets | ~33.000 bài |
| WikiQA-84k | HuggingFace Datasets | ~50.000 bài |
| Báo khác (Tuổi Trẻ, Dân Trí...) | Multi-source scraper | ~20.000 bài |

> Dữ liệu thô bị gitignore vì kích thước lớn. Tái tạo bằng `src/scraper.py` và `src/data_loader.py`.

---

## Cài Đặt & Chạy

### Yêu cầu

- Python 3.9+
- RAM 8 GB+ (khuyến nghị 16 GB)
- GPU CUDA (tuỳ chọn, tăng tốc embedding)

### Cài đặt

```bash
git clone https://github.com/khaooh12/123000882_NLP_Cuoiki
cd 123000882_NLP_Cuoiki

python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### Chạy demo Streamlit

```bash
streamlit run app.py
```

Upload file CSV (cần cột `text`/`title`/`description`/`content`) hoặc TXT (mỗi dòng 1 văn bản), chọn số chủ đề và nhấn **Phân Tích Ngay**.

### Chạy pipeline đầy đủ (nếu muốn retrain)

```bash
python src/scraper.py           # Thu thập dữ liệu
python src/preprocess.py        # Tiền xử lý
python src/embedding.py         # Tạo embeddings
python src/topic_model.py       # Huấn luyện mô hình
```

---

## Đánh Giá

| Metric | Ý nghĩa |
|--------|---------|
| **Coherence c_v** | Liên kết ngữ nghĩa giữa từ trong cùng topic — cao hơn là tốt hơn |
| **Silhouette Score** | Chất lượng phân cụm — [-1, 1], dương là tốt |
| **Topic Diversity** | Tỷ lệ từ khoá không trùng lặp giữa các topic |

---

## Stack Công Nghệ

| Thành phần | Thư viện |
|-----------|---------|
| Tiền xử lý NLP | `underthesea` |
| Embedding | `sentence-transformers` |
| Topic Modeling | `bertopic`, `gensim` |
| Dim. Reduction | `umap-learn`, `scikit-learn` |
| Visualization | `plotly`, `wordcloud`, `matplotlib` |
| Web Scraping | `requests`, `beautifulsoup4` |
| Demo | `streamlit` |

---

## Tài Nguyên

- [BERTopic](https://maartengr.github.io/BERTopic)
- [sentence-transformers](https://www.sbert.net)
- [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [underthesea](https://github.com/undertheseanlp/underthesea)
