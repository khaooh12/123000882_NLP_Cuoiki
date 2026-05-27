# 🗺️ Phân Cụm Chủ Đề Văn Bản Tiếng Việt — NLP Cuối Khoá

> **MSSV:** 123000882
> **Môn học:** Xử Lý Ngôn Ngữ Tự Nhiên (NLP)
> **Đề tài:** Phân cụm chủ đề văn bản tiếng Việt (Vietnamese Topic Modeling & Clustering)

---

## 📌 Mô Tả Đề Tài

### Bài toán

Cho một tập hợp lớn các văn bản tiếng Việt (bài báo, bình luận, tài liệu...) **không có nhãn sẵn**, hệ thống tự động:

1. **Khám phá** các chủ đề tiềm ẩn trong tập văn bản (unsupervised)
2. **Gán nhãn chủ đề** cho từng văn bản
3. **Trực quan hóa** phân bố cụm và từ khoá đặc trưng mỗi chủ đề

```
Tập văn bản thô  ──►  Embedding  ──►  Clustering  ──►  Chủ đề + Visualization
(không có nhãn)       (SBERT)          (UMAP+HDBSCAN)   (scatter plot, word cloud)
```

### Ví dụ minh hoạ

| Văn bản đầu vào | Chủ đề được khám phá |
|-----------------|----------------------|
| *"VNIndex tăng 15 điểm, nhà đầu tư mua mạnh cổ phiếu ngân hàng"* | 📈 Tài chính — Chứng khoán |
| *"Đội tuyển Việt Nam thắng 2-0 trước Thái Lan tại AFF Cup"* | ⚽ Thể thao |
| *"Bộ GD-ĐT công bố điểm chuẩn đại học năm 2025"* | 🎓 Giáo dục |
| *"Áp thấp nhiệt đới mạnh lên thành bão số 3, đổ bộ vào miền Trung"* | 🌧️ Thời tiết — Môi trường |

### Ứng dụng thực tế

- **Nghiên cứu thị trường:** Phân tích xu hướng dư luận từ mạng xã hội (social listening)
- **Toà soạn báo:** Tự động phân nhóm hàng nghìn bài báo theo chủ đề mà không cần gán nhãn thủ công
- **Chăm sóc khách hàng:** Phân nhóm ticket hỗ trợ để phát hiện vấn đề phổ biến
- **Khoa học:** Khám phá xu hướng nghiên cứu từ tập hợp abstract bài báo

---

## 🗂️ Dữ Liệu

### Nguồn dữ liệu (toàn bộ miễn phí)

| Nguồn | Phương pháp thu thập | Số lượng dự kiến | Chủ đề |
|-------|---------------------|-----------------|--------|
| **VnExpress** | Web scraping (`requests` + `BeautifulSoup`) | ~5.000–10.000 bài | Đa dạng (Thời sự, Kinh doanh, Thể thao, Giáo dục, Sức khỏe...) |
| **20 Newsgroups** | `sklearn.datasets.fetch_20newsgroups` | 18.846 bài | Tiếng Anh — dùng để kiểm tra pipeline |
| **VNTC** (tùy chọn) | Hugging Face Datasets | ~33.000 bài | Phân loại báo tiếng Việt |

> **Kế hoạch scraping VnExpress:**
> - Scrape các chuyên mục: Thời sự, Kinh doanh, Thể thao, Giáo dục, Sức khỏe, Pháp luật, Du lịch
> - Lấy tiêu đề + mô tả ngắn (không cần toàn bộ bài)
> - Tuân thủ `robots.txt`, rate limit 1 request / giây

---

## 🛠️ Công Nghệ Sử Dụng

### Hai hướng tiếp cận — so sánh song song

#### Hướng 1: LDA (Latent Dirichlet Allocation) — Classical

```
TF-IDF Vectorizer  ──►  LDA (Gensim)  ──►  pyLDAvis (interactive)
```

| Thư viện | Vai trò |
|---------|---------|
| `gensim` | Train LDA model |
| `pyLDAvis` | Visualize chủ đề tương tác |
| `nltk` / `underthesea` | Tiền xử lý, stopwords |

#### Hướng 2: BERTopic — Modern (chủ đạo)

```
underthesea (tokenize)
       │
       ▼
sentence-transformers     ──►  Embedding vector (768-dim)
(paraphrase-multilingual)
       │
       ▼
UMAP (giảm chiều)          ──►  2D / 5D representation
       │
       ▼
HDBSCAN (clustering)       ──►  Cụm chủ đề
       │
       ▼
c-TF-IDF (top keywords)    ──►  Từ khoá đặc trưng mỗi chủ đề
       │
       ▼
Visualization              ──►  Scatter plot + Word cloud + Topic bar
```

### Stack đầy đủ

| Thành phần | Công cụ | Phiên bản |
|-----------|---------|-----------|
| **Tiền xử lý** | `underthesea` | ≥ 6.8 |
| **Embedding** | `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2` | ≥ 2.7 |
| **Topic Modeling** | `bertopic` | ≥ 0.16 |
| **Dim. Reduction** | `umap-learn` | ≥ 0.5 |
| **Clustering** | `hdbscan` | ≥ 0.8 |
| **Classical NLP** | `gensim` + `pyLDAvis` | ≥ 4.3 |
| **Visualization** | `plotly`, `wordcloud`, `matplotlib` | latest |
| **Web Scraping** | `requests`, `beautifulsoup4` | latest |
| **Demo App** | `streamlit` | ≥ 1.30 |
| **Deploy** | Streamlit Cloud | — |

> **Yêu cầu phần cứng:**
> - Model `paraphrase-multilingual-MiniLM-L12-v2` chỉ ~120 MB — fit tốt trong 4 GB VRAM (RTX 3050)
> - UMAP + HDBSCAN chạy trên CPU với 32 GB RAM: ổn cho 10.000 văn bản
> - Toàn bộ pipeline không cần fine-tune — chỉ inference → train nhanh

---

## 📁 Cấu Trúc Thư Mục

```
123000882_NLP_Cuoiki/
│
├── data/
│   ├── raw/
│   │   ├── vnexpress_articles.csv      # Dữ liệu scrape từ VnExpress (700 bài)
│   │   ├── vntc_articles.csv           # VNTC-10Topics — HuggingFace (VI)
│   │   ├── newsgroups_articles.csv     # 20 Newsgroups — sklearn (EN)
│   │   ├── wiki_qa_articles.csv        # WikiQA-84k — HuggingFace (VI)
│   │   ├── wiki_mini_articles.csv      # wiki-mini-corpus — HuggingFace (VI)
│   │   ├── wiki_dump_articles.csv      # wiki-dump-cleaned — HuggingFace (VI)
│   │   └── all_articles.csv            # Gộp toàn bộ nguồn
│   └── processed/
│       ├── cleaned_articles.csv        # Sau tiền xử lý
│       └── embeddings.npy              # Vector SBERT (cache để tái sử dụng)
│
├── src/                                # ★ Module pipeline chính
│   ├── scraper.py          # Bước 1: Scrape VnExpress (real + dummy data)
│   ├── data_loader.py      # Bước 1b: Tải thêm từ HuggingFace / Kaggle
│   ├── preprocess.py       # Bước 2: Tiền xử lý, chuẩn hóa tiếng Việt
│   ├── embedding.py        # Bước 3: Tạo SBERT sentence embeddings
│   ├── topic_model.py      # Bước 4: Huấn luyện LDA & BERTopic (+ Fallback KMeans)
│   └── visualize.py        # Hàm vẽ biểu đồ scatter, word cloud, bar chart
│
├── models/
│   ├── lda_model/          # Gensim LDA: dictionary, corpus, model, pyldavis HTML
│   ├── bertopic_model/     # BERTopic chính thức hoặc Fallback KMeans
│   └── metrics.json        # Coherence Score (c_v) của cả hai mô hình
│
├── scripts/                            # Script tiện ích (không thuộc pipeline)
│   ├── scrape_test_set.py  # Cào tập kiểm thử từ VietnamNet
│   ├── test_model.py       # Đánh giá mô hình trên tập kiểm thử
│   └── evaluate_dataset.py # Phân tích thống kê toàn bộ dataset
│
├── scratch/                            # Script thử nghiệm / debug
│   ├── check_models.py     # Kiểm tra sự tồn tại file model (debug)
│   └── train_kmeans.py     # Retrain thủ công Fallback KMeans
│
├── app.py                  # ★ Streamlit demo app (entry point)
├── requirements.txt        # Khai báo thư viện
├── README.md               # Tài liệu dự án
├── SETUP.md                # Hướng dẫn cài đặt môi trường
└── DataStructure.md        # Ghi chú cấu trúc dữ liệu CSV
```

---

## 🗓️ Kế Hoạch Triển Khai

### Thu thập & Khám phá dữ liệu

- [ ] Viết scraper cho VnExpress
  - Scrape 7 chuyên mục × ~700–1.000 bài = ~5.000–7.000 bài
  - Lưu vào `data/raw/vnexpress_articles.csv` (url, title, description, category, date)
- [ ] EDA (Exploratory Data Analysis)
  - Phân phối bài theo chuyên mục, độ dài văn bản, từ phổ biến nhất
  - Thống kê missing values, duplicate
- [ ] Tiền xử lý văn bản
  - Chuẩn hóa Unicode, loại ký tự đặc biệt, URL, số điện thoại
  - Tách từ bằng `underthesea`
  - Loại stopwords tiếng Việt
  - Ghép title + description thành một đoạn

---

### Xây dựng mô hình

- [ ] LDA baseline
  - TF-IDF vectorize → LDA (Gensim, k=7–10 topics)
  - Đánh giá: Coherence Score (c_v, u_mass)
  - Visualize bằng `pyLDAvis`
- [ ] BERTopic (mô hình chính)
  - Tạo embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (batch inference trên GPU)
  - Cache embeddings ra file `.npy`
  - UMAP giảm chiều (n_components=5 cho clustering, n_components=2 cho visualize)
  - HDBSCAN clustering (min_cluster_size=15)
  - c-TF-IDF trích xuất top-10 từ khoá mỗi topic
- [ ] Đánh giá & điều chỉnh
  - So sánh LDA vs BERTopic: Coherence, Diversity, số topic tìm được
  - Thử nghiệm với số topic khác nhau (guided topic modeling)
  - Phân tích topic -1 (outlier) trong HDBSCAN

---

### Xây dựng Demo Streamlit

- [ ] **Tab 1 — Phân tích văn bản mới:**
  - Người dùng nhập 1 đoạn văn hoặc upload file `.txt` / `.csv`
  - Hệ thống dự đoán chủ đề gần nhất + độ tương đồng (cosine similarity với centroid)
  - Hiển thị top-5 văn bản tương tự trong tập dữ liệu

- [ ] **Tab 2 — Khám phá toàn bộ tập dữ liệu:**
  - Scatter plot 2D (UMAP) — mỗi điểm là 1 bài báo, màu theo topic
  - Hover để xem tiêu đề bài báo
  - Word cloud cho từng chủ đề (chọn qua dropdown)
  - Biểu đồ bar: số bài viết theo từng chủ đề

- [ ] **Tab 3 — So sánh LDA vs BERTopic:**
  - Hiển thị từ khoá đặc trưng của mỗi phương pháp
  - pyLDAvis embed vào Streamlit (via `components.html`)
  - Bảng đánh giá Coherence Score

---

### Hoàn thiện & Deploy

- [ ] Tối ưu performance: cache embeddings, lazy load model
- [ ] Viết docstring, clean code
- [ ] Deploy lên **Streamlit Cloud**
- [ ] Viết báo cáo cuối khoá (phương pháp, kết quả, nhận xét, hướng phát triển)
- [ ] Chuẩn bị slide thuyết trình (~10–12 slide)
- [ ] Test toàn bộ pipeline end-to-end trên Streamlit Cloud

---

## 📊 Metrics Đánh Giá

Vì bài toán **unsupervised** (không có nhãn ground truth), đánh giá qua:

| Metric | Mô tả | Công cụ |
|--------|-------|---------|
| **Coherence Score (c_v)** | Mức độ liên kết ngữ nghĩa giữa các từ trong cùng topic | `gensim.models.coherencemodel` |
| **Topic Diversity** | Tỷ lệ từ khoá không trùng lặp giữa các topic | Tính thủ công |
| **Silhouette Score** | Chất lượng phân cụm (compact + separated) | `sklearn.metrics` |
| **Kiểm tra định tính** | Con người đánh giá sự hợp lý của các chủ đề | Manual review |
| **Validation gián tiếp** | So sánh topic được tìm với chuyên mục gốc VnExpress | Accuracy mapping |

---

## 📈 Kết Quả Kỳ Vọng

| Phương pháp | Số topic | Coherence (c_v) | Thời gian train |
|-------------|----------|----------------|----------------|
| LDA | 7–10 | ~0.45–0.55 | < 5 phút |
| BERTopic | tự động (10–20) | ~0.55–0.70 | ~10–20 phút (embedding) |

> Kết quả thực tế sẽ được cập nhật sau khi train xong.

---

## ⚙️ Cài Đặt & Chạy

### Yêu cầu hệ thống

- Python 3.9+
- GPU CUDA (tùy chọn — dùng để tăng tốc embedding inference)
- RAM: 8 GB+ (khuyến nghị 16 GB+)

### Cài đặt

```bash
# Clone project
git clone <repo-url>
cd 123000882_NLP_Cuoiki

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Cài đặt thư viện
pip install -r requirements.txt
```

### Chạy pipeline

```bash
# Bước 1: Thu thập dữ liệu
python src/scraper.py

# Bước 2: Tiền xử lý
python src/preprocess.py

# Bước 3: Tạo embeddings (tốn thời gian nhất)
python src/embedding.py

# Bước 4: Train BERTopic
python src/topic_model.py

# Bước 5: Chạy Streamlit demo
streamlit run app.py
```

### `requirements.txt`

```
# Core NLP
underthesea>=6.8.0
sentence-transformers>=2.7.0
bertopic>=0.16.0
gensim>=4.3.0
umap-learn>=0.5.6
hdbscan>=0.8.33

# Machine Learning
torch>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0

# Visualization
plotly>=5.17.0
wordcloud>=1.9.3
matplotlib>=3.7.0
pyLDAvis>=3.4.0

# Web scraping
requests>=2.31.0
beautifulsoup4>=4.12.0

# App
streamlit>=1.30.0
```

---

## 🔗 Tài Nguyên Tham Khảo

| Tài nguyên | Link |
|-----------|------|
| BERTopic (paper + docs) | https://maartengr.github.io/BERTopic |
| sentence-transformers | https://www.sbert.net |
| paraphrase-multilingual-MiniLM-L12-v2 | https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| underthesea | https://github.com/undertheseanlp/underthesea |
| Gensim LDA | https://radimrehurek.com/gensim/models/ldamodel.html |
| pyLDAvis | https://github.com/bmabey/pyLDAvis |
| Streamlit Cloud | https://streamlit.io/cloud |

---

## 👤 Thông Tin Sinh Viên

| | |
|--|--|
| **MSSV** | 123000882 |
| **Môn học** | Xử Lý Ngôn Ngữ Tự Nhiên |
| **Đề tài** | Phân Cụm Chủ Đề Văn Bản Tiếng Việt |
| **Phương pháp chính** | BERTopic (SBERT + UMAP + HDBSCAN) |
| **Dữ liệu** | VnExpress scraping (~5.000–10.000 bài) |
| **Deploy** | Streamlit Cloud |
